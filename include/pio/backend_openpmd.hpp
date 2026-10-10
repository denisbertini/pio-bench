// pio/backend_openpmd.hpp -- openPMD-api backend: production PIC checkpoint
// semantics on the two engines, as --backend pmd_hdf5 | pmd_adios2.
//
// Why this backend exists: openPMD-api is the de-facto checkpoint layer of
// the PIC/exa-scale world (PIConGPU, WarpX, PiCoMo...). It re-expresses the
// same payload as the raw backends but through the STANDARD data model:
//   field      -> meshes/"field"/"rho"   float64, global extent,
//                 chunk = per-rank local block (the production pattern:
//                 without explicit chunks, HDF5 defaults to whole-dataset
//                 chunks and multi-rank writes serialize).
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
// Hint pass-through: openPMD engines take JSON options, not MPI_Info. The
// harness actuation channel for this backend is openPMD's own option layer
// (Series::setOptions); wiring an env channel through it needs the exact
// option schema of the image's openPMD version -- deliberate follow-up,
// NOT silently guessed here. Until then these backends measure openPMD's
// automatic choices, bit-exactly verified like every other backend.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
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
/// Current API (openPMD >= 0.15 / dev): backend chosen by JSON options,
/// ctor is Series(path, access, comm, options). The CMake probe picks the
/// ctor form; PIO_PMD_LEGACY_CTOR selects the old positional-engine one.
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
/// engine string and the stamped name/extension (concept requirements).
class OpenPmdBase {
public:
    OpenPmdBase(MPI_Comm comm, const IoSettings& s)
        : comm_(comm), settings_(s) {}

    WriteSplit write(const std::filesystem::path& path, const Domain& d,
                     const Field3d<double>& field,
                     const ParticleSet& parts) const {
        namespace om = ::openPMD;
        WriteSplit split;

        om::Series series = openpmd_detail::make_series(
            path, backend_, om::Access::Create, comm_);
        auto iteration = series.iterations[0];

        const om::Shape gsz{d.global[0], d.global[1], d.global[2]};
        const om::Offset st{d.start[0], d.start[1], d.start[2]};
        const om::Shape lsz{d.local[0], d.local[1], d.local[2]};

        // ---- fields: standard mesh record, chunk = local block ----------
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
        mesh.setGeometry(om::Geometry::cartesian);
        auto rho = mesh["rho"];
        rho.resetDataset(om::Dataset(om::determineType<double>(), gsz));
        rho.setChunkSize(lsz);
        rho.storeChunk(fbuf.data(), st, lsz);
        series.flush();
        split.field_seconds = w.elapsed();

        // ---- particles: standard SoA species -----------------------------
        // Type-safe traversal: species[record] is a PatchRecord (writable
        // directly for SCALAR records like mass/id); [record][comp] is the
        // PatchRecordComponent leaf. Generic lambda covers both shapes
        // without naming the (version-sensitive) classes.
        w.restart();
        const std::size_t n = parts.local_count();
        if (n > 0) {
            auto species = iteration.particles["electrons"];
            const om::Shape gcount{parts.global_count()};
            const om::Offset roff{parts.rank_offset()};
            const om::Shape lcount{n};
            const Particle* p = parts.data();

            for (const auto& m : openpmd_detail::members) {
                std::vector<double> col(n);
                for (std::size_t k = 0; k < n; ++k)
                    col[k] = p[k].*(m.dm);
                auto rec = species[m.record];
                if (m.comp) {
                    auto ds = rec[m.comp];
                    ds.resetDataset(
                        om::Dataset(om::determineType<double>(), gcount));
                    ds.setChunkSize(lcount);
                    ds.storeChunk(col.data(), roff, lcount);
                } else {
                    rec.resetDataset(
                        om::Dataset(om::determineType<double>(), gcount));
                    rec.setChunkSize(lcount);
                    rec.storeChunk(col.data(), roff, lcount);
                }
            }
            // id: the standard unsigned-64 particle id record (scalar).
            std::vector<std::uint64_t> ids(n);
            for (std::size_t k = 0; k < n; ++k)
                ids[k] = p[k].id;
            auto idrec = species["id"];
            idrec.resetDataset(
                om::Dataset(om::determineType<std::uint64_t>(), gcount));
            idrec.setChunkSize(lcount);
            idrec.storeChunk(ids.data(), roff, lcount);
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
            path, backend_, om::Access::Read, comm_);
        auto iteration = series.iterations[0];
        auto rho = iteration.meshes["field"]["rho"];

        const om::Offset st{d.start[0], d.start[1], d.start[2]};
        const om::Shape lsz{d.local[0], d.local[1], d.local[2]};
        const std::size_t lx = d.local[0], ly = d.local[1], lz = d.local[2];
        std::vector<double> fbuf(lx * ly * lz);
        rho.loadChunk(fbuf.data(), st, lsz);
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
            path, backend_, om::Access::Read, comm_);
        auto iteration = series.iterations[0];
        auto species = iteration.particles["electrons"];

        const om::Offset roff{src.rank_offset()};
        const om::Shape lcount{n};

        std::vector<std::shared_ptr<const double>> cols;
        cols.reserve(std::size(openpmd_detail::members));
        for (const auto& m : openpmd_detail::members) {
            auto rec = species[m.record];
            if (m.comp) {
                auto ds = rec[m.comp];
                cols.push_back(ds.loadChunk<double>(roff, lcount));
            } else {
                cols.push_back(rec.loadChunk<double>(roff, lcount));
            }
        }
        auto ids = species["id"].loadChunk<std::uint64_t>(roff, lcount);
        series.flush();

        for (std::size_t k = 0; k < n; ++k) {
            Particle& p = out[k];
            for (std::size_t c = 0; c < std::size(openpmd_detail::members); ++c)
                p.*(openpmd_detail::members[c].dm) = cols[c].get()[k];
            p.id = ids.get()[k];
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
        : OpenPmdBase(comm, s), backend_("hdf5") {}
};

class OpenPmdAdios2Backend : public OpenPmdBase {
public:
    static constexpr std::string_view name = "pmd_adios2";
    static constexpr std::string_view extension = ""; // openPMD appends .bp
    static constexpr bool available = true;
    OpenPmdAdios2Backend(MPI_Comm comm, const IoSettings& s)
        : OpenPmdBase(comm, s), backend_("adios2") {}
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
