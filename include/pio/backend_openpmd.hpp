// pio/backend_openpmd.hpp -- openPMD-api backend: production PIC checkpoint
// semantics on the two engines, as --backend pmd_hdf5 | pmd_adios2.
//
// Why this backend exists: openPMD-api is the de-facto checkpoint layer of
// the PIC/exa-scale world (PIConGPU, WarpX, PiCoMo...). It re-expresses the
// same payload as the raw backends but through the STANDARD data model:
//   field      -> meshes/"field"/"rho"   float64, global extent; the HDF5
//                 chunk layout follows the first written block (the per-rank
//                 local subbrick), i.e. the production write pattern.
//   particles  -> particles/"electrons"  standard SoA species:
//                 position(x,y,z), momentum(x,y,z), mass, id(uint64)
//                 -- the AoS->SoA transpose is exactly the packing that
//                 production openPMD writers pay; it is timed INSIDE the
//                 particle window (same philosophy as the hdf5 prepack).
//
// CANONICAL IMPLEMENTATION: the write path mirrors openPMD's own parallel
// benchmark example (examples/8a_benchmark_write_parallel.cpp) and the
// particle example (examples/3b_write_resizable_particles.cpp) at the exact
// 0.17.1 tag, call for call:
//   * OWNING buffers: every storeChunk() uses the shared_ptr overload -- the
//     createData() pattern of example 8a.  openPMD holds the buffer until
//     the flush consumes it, so a dangling-buffer flush is impossible BY
//     CONSTRUCTION.  (The zero-copy container overload storeChunk(vec&, ...)
//     keeps only a raw pointer -- fine while buffers outlive the flush, but
//     it is the classic openPMD user foot-gun and buys nothing here.)
//   * SCALAR particle records are written DIRECTLY ON THE RECORD:
//     currSpecies["id"].resetDataset(ds); currSpecies["id"].storeChunk(...)
//     This works because 0.17's BaseRecord<T> inherits BOTH
//     Container<T> AND T itself ("if the record is a scalar record, it
//     directly acts as a record component" -- BaseRecord.hpp, 0.17.1).
//     The previous implementation reached mass/id through a separate
//     RecordComponent(Record) view object -- a real-library capability,
//     but NOT the canonical path, and the only non-canonical API in the
//     write path (it is retired here).
//   * positionOffset(x,y,z) are STANDARD species records (openPMD particle
//     spec); like examples 8a/3b they are makeConstant(0.0) -- metadata,
//     zero data bytes.
//   * Flush discipline: the canonical write is ONE flush at
//     iteration.close() ("The iteration's content will be flushed
//     automatically" -- example 3b).  That is the DEFAULT.  --pmd-split
//     opts into two incremental flushes for an honest field/particle phase
//     split:  [pack+store field | flush] = field_seconds
//              [pack+store parts | flush] = particle_seconds
//              iteration.close() (metadata only)
//     With the split off, both phases report 0 and the wall total in
//     benchmark.hpp remains exact.
//
// API contract: written against openPMD 0.17.x (verified against the exact
// tag sources of 0.17.1): Series(path, access, comm, options-JSON) with the
// backend selected via {"backend":"hdf5"|"adios2"}; Access::CREATE /
// READ_ONLY; determineDatatype<T>(); Offset/Extent; storeChunk via
// std::shared_ptr overloads (shared_ptr<T> and shared_ptr<T[]>); raw-buffer
// reads via loadChunkRaw(T*, ...) (the buffer must stay alive until flush;
// read buffers below are function-scope); scalar records used as components
// directly on the Record (BaseRecord<T> : Container<T>, T).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
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

/// PIO_PMD_DEBUG=1: dump every storeChunk parameter before enqueue so a
/// crash run is self-describing (dataset, offset, extent, buffer pointer).
inline bool pmd_debug() {
    static const bool on = [] {
        const char* e = std::getenv("PIO_PMD_DEBUG");
        return e && e[0] == '1';
    }();
    return on;
}
inline void dbg_store(const char* what, const std::vector<uint64_t>& off,
                      const std::vector<uint64_t>& ext, const void* ptr) {
    if (!pmd_debug()) return;
    fprintf(stderr,
            "[pmd-dbg] store %-28s off={%llu,%llu,%llu} ext={%llu,%llu,%llu} "
            "buf=%p\n",
            what, (unsigned long long)(off.size() > 0 ? off[0] : 0),
            (unsigned long long)(off.size() > 1 ? off[1] : 0),
            (unsigned long long)(off.size() > 2 ? off[2] : 0),
            (unsigned long long)(ext.size() > 0 ? ext[0] : 0),
            (unsigned long long)(ext.size() > 1 ? ext[1] : 0),
            (unsigned long long)(ext.size() > 2 ? ext[2] : 0), ptr);
}
inline void dbg_flush(const char* phase) {
    if (pmd_debug()) fprintf(stderr, "[pmd-dbg] FLUSH >>> %s\n", phase);
}

/// Standard-name member table: AoS member -> (record, component).
/// Scalar records carry comp == nullptr: BaseRecord IS-A RecordComponent,
/// so they are written directly on the Record (example 8a: currSpecies["id"]).
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
    // Give openPMD the resolved filename outright (base + engine suffix):
    // an extension-less name works via the backend JSON but makes ADIOS2
    // emit one "No file ending specified" warning per rank per Series --
    // and leaves the on-disk name implicit. Explicit is quieter and clearer.
    //
    // ADIOS2 runs file-based iteration encoding (see write()), which the
    // API requires to carry the %T iteration-expansion pattern in the
    // name: "<base>_%T.bp"  (the underscore is openPMD's own advice --
    // digits directly before %T make the on-disk iteration-number parsing
    // ambiguous, e.g. prefix "chkpt_000001" + iteration 0 = "chkpt_0000010").
    // %T expands to the iteration index (always 0 here -- one iteration per
    // Series), so the on-disk series directory keeps the checkpoint name
    // intact.  The read side passes the same pattern; openPMD scans it and
    // resolves the existing iterations.
    const bool adios2 = std::string_view(backend) == "adios2";
    const std::string name =
        base.string() + (adios2 ? "_%T.bp" : ".h5");
#ifdef PIO_PMD_LEGACY_CTOR
    const std::string engine =
        std::string_view(backend) == "adios2" ? "BP5" : "HDF5";
    (void)std::getenv("PIOB_PMD_OPTS"); // passthrough needs the 0.17 options
                                        // JSON; ignored on the legacy ctor
    return ::openPMD::Series(name, at, engine, comm);
#else
    // Series open options: backend selector + optional raw passthrough
    // PIOB_PMD_OPTS='{"hdf5":{"vfd":{"type":"subfiling",...}}}' -- the
    // escape hatch for every openPMD open-time knob (VFD/subfiling,
    // dataset defaults, collective metadata...) without code churn.
    // Must be a JSON OBJECT; merged after the backend key (which the
    // passthrough may not override -- last-wins JSON parsing would let a
    // stray "backend" key flip engines mid-benchmark).
    std::string opts = std::string("{\"backend\":\"") + backend + "\"";
    if (const char* extra = std::getenv("PIOB_PMD_OPTS"); extra && *extra) {
        const std::string e = extra;
        const std::size_t b = e.find_first_not_of(" \t\n\r"),
                        f = e.find_last_not_of(" \t\n\r");
        if (b == std::string::npos)
            ; // whitespace-only: treat as unset
        else if (e[b] != '{' || e[f] != '}' || f <= b)
            throw std::runtime_error(
                "PIOB_PMD_OPTS must be a JSON object, got: " + e);
        else {
            const std::string inner = e.substr(b + 1, f - b - 1);
            if (inner.find_first_not_of(" \t\n\r") != std::string::npos) {
                if (inner.find("\"backend\"") != std::string::npos)
                    throw std::runtime_error(
                        "PIOB_PMD_OPTS must not set \"backend\" (owned by "
                        "--backend): " + e);
                opts += "," + inner;
            }
        }
    }
    opts += "}";
    return ::openPMD::Series(name, at, comm, opts);
#endif
}

/// RANK-ALIGNED HDF5 CHUNKS (production pattern, PIConGPU/WarpX).
///
/// Without explicit chunks the 0.17.1 HDF5 handler computes "auto" chunks
/// capped at 4 MiB (getOptimalChunkDims, HDF5Auxiliary.cpp): a 128 MiB
/// per-rank brick becomes ~64 chunks and a 32 MiB particle column 8, each
/// first-touch an independently-locked chunk allocation -> aggregate
/// throughput collapses (~10 MiB/s/rank measured on Lustre/Virgo).  Aligning
/// the chunk with the per-rank box makes every rank write ONE chunk of its
/// data: contiguous multi-MiB extents, minimal allocation traffic, and no
/// chunk shared between ranks (the failure mode openPMD's ParallelIOTest
/// warns about for independent writes).
///
/// Per-dataset option format is openPMD 0.17's own, from
/// test/ParallelIOTest.cpp:  {"hdf5":{"dataset":{"chunks":[...]}}}
/// (case-insensitive keys).  ADIOS2 ignores it -- BP5 aggregates per
/// writer on its own -- so the hdf5-engine gate is deliberate.
///
/// The chunk extent MUST be identical on all ranks (it defines the shared
/// dataset); local boxes are uniform by construction and the particle count
/// is MIN-reduced to guarantee it.  Returns "{}" (= library auto) if
/// disabled or if the aligned chunk would exceed the size cap.
inline constexpr std::uint64_t kChunkBytesCap = 1ull << 30;  // 1 GiB

inline std::string chunk_opts(const char* backend,
                              const std::vector<std::uint64_t>& chunk) {
    if (std::string_view(backend) != "hdf5" || chunk.empty()) return "{}";
    std::string s = R"({"hdf5":{"dataset":{"chunks":[)";
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(chunk[i]);
    }
    s += "]}}}";  // array + dataset + hdf5 + root objects
    return s;
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
        if (std::string_view(backend_) == "adios2") {
            // openPMD's startup warning, verbatim: "Use of group-based
            // encoding in ADIOS2 is discouraged as it can lead to drastic
            // performance issues" -- one atomic logical write at close and
            // metadata that grows per step.  File-based encoding is the
            // recommended mode (it is what PIConGPU ships): one ADIOS2
            // file per iteration, each closed atomically.  With one
            // iteration per Series here, that is exactly the checkpoint
            // pattern.  (Read side auto-detects the encoding; no matching
            // change needed there.)
            series.setIterationEncoding(om::IterationEncoding::fileBased);
        }
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
        // openPMD writers make; timed on purpose).  OWNING shared buffer,
        // canonical createData()/storeChunk(shared_ptr) pattern: ownership
        // transfers to openPMD, the bytes cannot dangle at flush.
        const std::size_t lx = d.local[0], ly = d.local[1], lz = d.local[2];
        const std::size_t py = d.padded[1], pz = d.padded[2];
        std::shared_ptr<double[]> fbuf(new double[lx * ly * lz]);
        for (std::size_t i = 0; i < lx; ++i)
            for (std::size_t j = 0; j < ly; ++j) {
                const double* src =
                    field.data() +
                    ((i + d.ghost) * py + (j + d.ghost)) * pz + d.ghost;
                std::memcpy(&fbuf[i * ly * lz + j * lz], src,
                            lz * sizeof(double));
            }

        auto mesh = iteration.meshes["field"];
        mesh.setGeometry(om::Mesh::Geometry::cartesian); // nested enum (0.17.x)
        auto rho = mesh["rho"];
        // Rank-aligned chunk: the whole per-rank brick is one chunk.
        std::vector<std::uint64_t> fchunk;
        if (settings_.pmd_rank_chunks && lx * ly * lz * sizeof(double) <=
                                            openpmd_detail::kChunkBytesCap)
            fchunk = {lx, ly, lz};
        rho.resetDataset(
            om::Dataset(om::determineDatatype<double>(), gsz,
                        openpmd_detail::chunk_opts(backend_, fchunk)));
        rho.storeChunk(fbuf, st, lsz); // shared_ptr overload: owning
        openpmd_detail::dbg_store("field/rho", st, lsz, fbuf.get());
        if (settings_.pmd_split) {
            openpmd_detail::dbg_flush("1/2: field");
            series.flush();
            split.field_seconds = w.elapsed();
        }

        // ---- particles: standard SoA species ------------------------------
        w.restart();
        const std::size_t n = parts.local_count();

        // Rank-aligned particle chunks: each rank's column is one chunk.
        // The chunk extent defines the shared dataset, so it is the MIN
        // of the per-rank counts (identical on every rank by reduction).
        // NOTE: collective -- must stay outside any per-rank n>0 guard.
        std::uint64_t pcmin = 0;
        const std::uint64_t pc = static_cast<std::uint64_t>(n);
        MPI_Allreduce(&pc, &pcmin, 1, MPI_UINT64_T, MPI_MIN, comm_);
        std::vector<std::uint64_t> pchunk;
        if (settings_.pmd_rank_chunks && pcmin > 0 &&
            pcmin <= parts.global_count() &&
            pcmin * sizeof(double) <= openpmd_detail::kChunkBytesCap)
            pchunk = {pcmin};
        const std::string popts =
            openpmd_detail::chunk_opts(backend_, pchunk);

        if (n > 0) {
            auto species = iteration.particles["electrons"];
            const om::Extent gcount{parts.global_count()};
            const om::Offset roff{parts.rank_offset()};
            const om::Extent lcount{n};
            const Particle* p = parts.data();

            constexpr std::size_t ncol =
                std::size(openpmd_detail::members);
            for (std::size_t c = 0; c < ncol; ++c) {
                const auto& m = openpmd_detail::members[c];
                // Column packed straight into an OWNING buffer; storeChunk()
                // hands it to openPMD, which keeps it alive until the flush.
                std::shared_ptr<double[]> col(new double[n]);
                double* raw = col.get();
                for (std::size_t k = 0; k < n; ++k)
                    col[k] = p[k].*(m.dm);

                if (m.comp) {
                    auto comp = species[m.record][m.comp];
                    comp.resetDataset(om::Dataset(
                        om::determineDatatype<double>(), gcount, popts));
                    comp.storeChunk(std::move(col), roff, lcount);
                } else {
                    // Canonical scalar record (example 8a): BaseRecord
                    // inherits RecordComponent itself, so the Record IS the
                    // dataset -- reset/store directly, no detached view.
                    auto rec = species[m.record];
                    rec.resetDataset(om::Dataset(
                        om::determineDatatype<double>(), gcount, popts));
                    rec.storeChunk(std::move(col), roff, lcount);
                }
                openpmd_detail::dbg_store(
                    (std::string(m.record) + (m.comp ? "/" : "") +
                     (m.comp ? m.comp : " [scalar]"))
                        .c_str(),
                    roff, lcount, raw);
            }
            // id: the standard unsigned-64 particle id record (scalar).
            std::shared_ptr<std::uint64_t[]> ids(new std::uint64_t[n]);
            std::uint64_t* raw_ids = ids.get();
            for (std::size_t k = 0; k < n; ++k)
                ids[k] = p[k].id;
            auto idrec = species["id"];
            idrec.resetDataset(om::Dataset(
                om::determineDatatype<std::uint64_t>(), gcount, popts));
            idrec.storeChunk(std::move(ids), roff, lcount);
            openpmd_detail::dbg_store("id [scalar]", roff, lcount, raw_ids);

            // positionOffset: standard-mandated species records (openPMD
            // particle spec).  Examples 8a/3b: resetDataset + makeConstant
            // -- pure metadata, zero data bytes on disk.
            static constexpr const char* poff_comp[3] = {"x", "y", "z"};
            for (const char* cc : poff_comp) {
                auto po = species["positionOffset"][cc];
                po.resetDataset(
                    om::Dataset(om::determineDatatype<double>(), gcount));
                po.makeConstant(0.0);
            }
        }
        if (settings_.pmd_split) {
            openpmd_detail::dbg_flush("2/2: particles");
            series.flush();
            split.particle_seconds = w.elapsed();
        }

        // Canonical finalization (examples 3b/8a): closing the iteration
        // flushes everything it still holds -- in the default mode this is
        // THE flush point.  With --pmd-split the bytes are already down and
        // this is metadata only.
        iteration.close();
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

        // loadChunkRaw is the raw counterpart of storeChunk: the caller's
        // buffer must outlive the flush -- cols/ids are function-scope.
        std::vector<std::vector<double>> cols;
        cols.reserve(std::size(openpmd_detail::members));
        for (const auto& m : openpmd_detail::members) {
            cols.emplace_back(n);
            if (m.comp) {
                auto comp = species[m.record][m.comp];
                comp.loadChunkRaw(cols.back().data(), roff, lcount);
            } else {
                // scalar record: the Record IS the component (see write).
                auto rec = species[m.record];
                rec.loadChunkRaw(cols.back().data(), roff, lcount);
            }
        }
        std::vector<std::uint64_t> ids(n);
        auto idrec = species["id"];
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
