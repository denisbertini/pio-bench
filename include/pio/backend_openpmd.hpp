// pio/backend_openpmd.hpp -- openPMD-api backend: production PIC checkpoint
// semantics on the two engines, as --backend pmd_hdf5 | pmd_adios2.
//
// Why this backend exists: openPMD-api is the de-facto checkpoint layer of
// the PIC/exa-scale world (PIConGPU, WarpX, PiCoMo...). It re-expresses the
// same payload as the raw backends but through the STANDARD data model:
//   field      -> meshes/"field"/"rho"   float64, global extent; the HDF5
//                 chunk layout follows the first written block (the per-rank
//                 local subbrick), i.e. the production write pattern.
//   particles  -> particles/"electrons"  standard SoA records:
//                 position(x,y,z), momentum(x,y,z), mass, id(uint64)
//                 -- the AoS->SoA transpose is exactly the packing that
//                 production openPMD writers pay; it is timed INSIDE the
//                 particle window (same philosophy as the hdf5 prepack).
//
// Timing honesty: openPMD store*() calls are LAZY -- bytes move at flush.
// The phase split therefore uses two incremental flushes within one
// iteration (supported by both the HDF5 and the ADIOS2/BP engines):
//   [pack+store field  | flush] = field_seconds
//   [pack+store parts  | flush] = particle_seconds
//   iteration.close()            (metadata only, outside both windows)
// The wall total in benchmark.hpp additionally covers Series create/close,
// so --fitness app never under-reports the openPMD bookkeeping.
//
// API contract: written against openPMD 0.17.x (verified against the exact
// tag sources of 0.17.1): Series(path, access, comm, options-JSON) with the
// backend selected via {"backend":"hdf5"|"adios2"}; Access::CREATE /
// READ_ONLY; determineDatatype<T>(); Offset/Extent (no Shape); raw-buffer
// I/O via the contiguous-container storeChunk(vector&, ...) and
// loadChunkRaw(T*, ...); scalar particle records reached through the
// RecordComponent(BaseRecord<RecordComponent>) slicing-safe view ctor.  //
// (0.17.1 fact: particle DATA records are plain Record; PatchRecord is
//  patch metadata only.)
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "pio/backends.hpp"

#ifdef PIO_HAVE_OPENPMD
#include <openPMD/openPMD.hpp>
#endif

namespace pio {

#ifdef PIO_HAVE_OPENPMD

namespace openpmd_detail {

/// Standard-name member table: AoS member -> (record, component).
/// Scalar records carry comp == nullptr (the record IS the dataset).
struct SoaMember {
    const char* record;
    const char* comp;
    double Particle::*dm;
};
inline constexpr SoaMember members[] = {
    {"position", "x", &Particle::x}, {"position", "y", &Particle::y},
    {"position", "z", &Particle::z}, {"momentum", "x", &Particle::px},
    {"momentum", "y", &Particle::py}, {"momentum", "z", &Particle::pz},
    {"mass", nullptr, &Particle::mass},
};

/// openPMD appends the engine suffix (.h5/.bp) when the base has none --
/// write and read must pass the identical base + backend to resolve it.
/// 0.17.x ctor: (path, access, comm, options), backend via JSON. The CMake
/// probe picks the ctor form; PIO_PMD_LEGACY_CTOR = pre-0.17 positional
/// engine-string form.
inline ::openPMD::Series make_series(const std::filesystem::path& base,
                                     const char* backend,
                                     ::openPMD::Access at, MPI_Comm comm) {
#ifdef PIO_PMD_LEGACY_CTOR
    const std::string engine =
        std::string_view(backend) == "adios2" ? "BP5" : "HDF5";
    return ::openPMD::Series(base.string(), at, engine, comm);
#else
    return ::openPMD::Series(base.string(), at, comm,
                             std::string("{\"backend\":\"") + backend + "\"}");
#endif
}

} // namespace openpmd_detail

/// Shared implementation; the two concrete backends differ only in the
/// backend string and the stamped name/extension (concept requirements).
class OpenPmdBase {
public:
    OpenPmdBase(MPI_Comm comm, const IoSettings& s, const char* backend)
        : comm_(comm), settings_(s), backend_(backend) {}

    WriteSplit write(const std::filesystem::path& path, const Domain& d,
                     const Field3d<double>& field,
                     const ParticleSet& parts) const {
        namespace om = ::openPMD;
        WriteSplit split;

        om::Series series = openpmd_detail::make_series(
            path, backend_, om::Access::CREATE, comm_);
        auto iteration = series.iterations[0];

        const om::Extent gsz{static_cast<std::uint64_t>(d.global[0]),
                             static_cast<std::uint64_t>(d.global[1]),
                             static_cast<std::uint64_t>(d.global[2])};
        const om::Offset st{static_cast<std::uint64_t>(d.start[0]),
                            static_cast<std::uint64_t>(d.start[1]),
                            static_cast<std::uint64_t>(d.start[2])};
        const om::Extent lsz{static_cast<std::uint64_t>(d.local[0]),
                             static_cast<std::uint64_t>(d.local[1]),
                             static_cast<std::uint64_t>(d.local[2])};

        // ---- fields: standard mesh record --------------------------------
        Stopwatch w;
        // Interior gather out of the ghost halo: contiguous staging buffer
        // (openPMD has no strided-memory store -- the same copy production
        // openPMD writers make; timed on purpose).
        const std::size_t lx = d.local[0], ly = d.local[1], lz = d.local[2];
        const std::size_t py = d.padded[1], pz = d.padded[2];
        std::vector<double> fbuf(lx * ly * lz);
        for (std::size_t i = 0; i < lx; ++i)
            for (std::size_t j = 0; j < ly; ++j) {
                const double* src =
                    field.data() +
                    ((i + d.ghost) * py + (j + d.ghost)) * pz + d.ghost;
                std::memcpy(&fbuf[(i * ly + j) * lz], src, lz * sizeof(double));
            }

        auto mesh = iteration.meshes["field"];
        mesh.setGeometry(om::Mesh::Geometry::cartesian); // nested enum (0.17.x)
        auto rho = mesh["rho"];
        rho.resetDataset(om::Dataset(om::determineDatatype<double>(), gsz));
        rho.storeChunk(fbuf, st, lsz); // contiguous-container overload
        series.flush();
        split.field_seconds = w.elapsed();

        // ---- particles: standard SoA species ------------------------------
        // Species records: [record][comp] is the data component; SCALAR
        // records (mass, id) are reached as component views via the
        // slicing-safe RecordComponent(BaseRecord) ctor.
        w.restart();
        const std::size_t n = parts.local_count();
        if (n > 0) {
            auto species = iteration.particles["electrons"];
            const om::Extent gcount{parts.global_count()};
            const om::Offset roff{parts.rank_offset()};
            const om::Extent lcount{n};
            const Particle* p = parts.data();

            for (const auto& m : openpmd_detail::members) {
                std::vector<double> col(n);
                for (std::size_t k = 0; k < n; ++k)
                    col[k] = p[k].*(m.dm);
                auto rec = species[m.record];
                if (m.comp) {
                    auto comp = rec[m.comp];
                    comp.resetDataset(
                        om::Dataset(om::determineDatatype<double>(), gcount));
                    comp.storeChunk(col, roff, lcount);
                } else {
                    om::RecordComponent comp(rec);
                    comp.resetDataset(
                        om::Dataset(om::determineDatatype<double>(), gcount));
                    comp.storeChunk(col, roff, lcount);
                }
            }
            // id: the standard unsigned-64 particle id record (scalar).
            std::vector<std::uint64_t> ids(n);
            for (std::size_t k = 0; k < n; ++k)
                ids[k] = p[k].id;
            om::RecordComponent idrec(species["id"]);
            idrec.resetDataset(
                om::Dataset(om::determineDatatype<std::uint64_t>(), gcount));
            idrec.storeChunk(ids, roff, lcount);
        }
        series.flush();
        split.particle_seconds = w.elapsed();

        iteration.close(); // metadata finalization, outside both windows
        return split;
    }

    void read_field(const std::filesystem::path& path, const Domain& d,
                    Field3d<double>& dst) const {
        namespace om = ::openPMD;
        om::Series series = openpmd_detail::make_series(
            path, backend_, om::Access::READ_ONLY, comm_);
        auto iteration = series.iterations[0];
        auto rho = iteration.meshes["field"]["rho"];

        const om::Offset st{static_cast<std::uint64_t>(d.start[0]),
                            static_cast<std::uint64_t>(d.start[1]),
                            static_cast<std::uint64_t>(d.start[2])};
        const om::Extent lsz{static_cast<std::uint64_t>(d.local[0]),
                             static_cast<std::uint64_t>(d.local[1]),
                             static_cast<std::uint64_t>(d.local[2])};
        const std::size_t lx = d.local[0], ly = d.local[1], lz = d.local[2];
        std::vector<double> fbuf(lx * ly * lz);
        rho.loadChunkRaw(fbuf.data(), st, lsz);
        series.flush(); // bytes now owned by fbuf

        const std::size_t py = d.padded[1], pz = d.padded[2];
        for (std::size_t i = 0; i < lx; ++i)
            for (std::size_t j = 0; j < ly; ++j)
                std::memcpy(
                    dst.data() + ((i + d.ghost) * py + (j + d.ghost)) * pz + d.ghost,
                    &fbuf[(i * ly + j) * lz], lz * sizeof(double));
    }

    void read_particles(const std::filesystem::path& path, const Domain& /*d*/,
                        const ParticleSet& src,
                        std::vector<Particle>& out) const {
        namespace om = ::openPMD;
        const std::size_t n = src.local_count();
        out.assign(n, Particle{});
        if (n == 0)
            return;

        om::Series series = openpmd_detail::make_series(
            path, backend_, om::Access::READ_ONLY, comm_);
        auto iteration = series.iterations[0];
        auto species = iteration.particles["electrons"];

        const om::Offset roff{src.rank_offset()};
        const om::Extent lcount{n};

        std::vector<std::vector<double>> cols;
        cols.reserve(std::size(openpmd_detail::members));
        for (const auto& m : openpmd_detail::members) {
            cols.emplace_back(n);
            auto rec = species[m.record];
            if (m.comp) {
                auto comp = rec[m.comp];
                comp.loadChunkRaw(cols.back().data(), roff, lcount);
            } else {
                om::RecordComponent comp(rec);
                comp.loadChunkRaw(cols.back().data(), roff, lcount);
            }
        }
        std::vector<std::uint64_t> ids(n);
        om::RecordComponent idrec(species["id"]);
        idrec.loadChunkRaw(ids.data(), roff, lcount);
        series.flush();

        for (std::size_t k = 0; k < n; ++k) {
            Particle& p = out[k];
            for (std::size_t c = 0; c < std::size(openpmd_detail::members); ++c)
                p.*(openpmd_detail::members[c].dm) = cols[c][k];
            p.id = ids[k];
        }
    }

protected:
    MPI_Comm comm_;
    IoSettings settings_;
    const char* backend_;  // openPMD backend name: "hdf5" | "adios2"
};

class OpenPmdHdf5Backend : public OpenPmdBase {
public:
    static constexpr std::string_view name = "pmd_hdf5";
    static constexpr std::string_view extension = ""; // openPMD appends .h5
    static constexpr bool available = true;
    OpenPmdHdf5Backend(MPI_Comm comm, const IoSettings& s)
        : OpenPmdBase(comm, s, "hdf5") {}
};

class OpenPmdAdios2Backend : public OpenPmdBase {
public:
    static constexpr std::string_view name = "pmd_adios2";
    static constexpr std::string_view extension = ""; // openPMD appends .bp
    static constexpr bool available = true;
    OpenPmdAdios2Backend(MPI_Comm comm, const IoSettings& s)
        : OpenPmdBase(comm, s, "adios2") {}
};

#else // !PIO_HAVE_OPENPMD -- compile-time stubs keep the concept satisfied

class OpenPmdHdf5Backend {
public:
    static constexpr std::string_view name = "pmd_hdf5";
    static constexpr std::string_view extension = "";
    static constexpr bool available = false;
    OpenPmdHdf5Backend(MPI_Comm, const IoSettings&) {
        throw std::runtime_error("build without openPMD-api (FindopenPMD)");
    }
    WriteSplit write(const std::filesystem::path&, const Domain&,
                     const Field3d<double>&, const ParticleSet&) const {
        throw std::runtime_error("build without openPMD-api");
    }
    void read_field(const std::filesystem::path&, const Domain&,
                    Field3d<double>&) const {
        throw std::runtime_error("build without openPMD-api");
    }
    void read_particles(const std::filesystem::path&, const Domain&,
                        const ParticleSet&, std::vector<Particle>&) const {
        throw std::runtime_error("build without openPMD-api");
    }
};

class OpenPmdAdios2Backend {
public:
    static constexpr std::string_view name = "pmd_adios2";
    static constexpr std::string_view extension = "";
    static constexpr bool available = false;
    OpenPmdAdios2Backend(MPI_Comm, const IoSettings&) {
        throw std::runtime_error("build without openPMD-api (FindopenPMD)");
    }
    WriteSplit write(const std::filesystem::path&, const Domain&,
                     const Field3d<double>&, const ParticleSet&) const {
        throw std::runtime_error("build without openPMD-api");
    }
    void read_field(const std::filesystem::path&, const Domain&,
                    Field3d<double>&) const {
        throw std::runtime_error("build without openPMD-api");
    }
    void read_particles(const std::filesystem::path&, const Domain&,
                        const ParticleSet&, std::vector<Particle>&) const {
        throw std::runtime_error("build without openPMD-api");
    }
};

#endif

} // namespace pio
