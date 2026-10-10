// pio/backends.hpp -- backend concept + conditional aggregate.
//
// Modern replacement for the `m_io == CI_OMPI_TYPE_PIC_*` if-ladders:
// backends are POLICY types satisfying pio::IoBackend; the benchmark is a
// template over them and the concrete choice is made exactly once, in
// main(), via `if constexpr` over compile-time availability -- so a build
// without ADIOS2 doesn't even contain the ADIOS2 code path, and a wrong
// backend interface is a compile error, not a silent fall-through.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "pio/domain.hpp"
#include "pio/field.hpp"
#include "pio/mpi.hpp"
#include "pio/params.hpp"
#include "pio/particles.hpp"
#include "pio/stats.hpp"

namespace pio {

/// I/O tuning knobs, translated per backend (ROMIO hints / ADIOS params).
struct IoSettings {
    std::optional<int> aggregators;
    std::optional<std::size_t> buffer_bytes;
    bool collective_buffering{true};
    bool h5_prepack{true};  // hdf5: gather members before H5Dwrite (cf. --h5-strided)
    bool adi_stats{true};   // adios2: BP5 per-block stats (cf. --no-adi-stats)
    bool pmd_split{false};  // pmd_*: two incremental flushes -> real phase split
                            // (default: canonical single flush at
                            // iteration.close(), cf. --pmd-split)
    bool pmd_rank_chunks{true}; // pmd_hdf5: chunk datasets at the per-rank box
                                // (production pattern; false = openPMD "auto"
                                // 4 MiB-capped chunks, cf. --pmd-auto-chunks)

    static IoSettings from(const BenchConfig& c) {
        return IoSettings{c.aggregators, c.buffer_bytes, c.collective_buffering,
                          c.h5_prepack, c.adi_stats, c.pmd_split,
                          c.pmd_rank_chunks};
    }

    /// ROMIO hint set shared by the mpiio and hdf5 (H5Pset_fapl_mpio) paths.
    ///
    /// PASS-THROUGH semantics (required by tuning harnesses, cf.
    /// mpiio_evolve): MPI_Info hints OVERRIDE ROMIO's ROMIO_HINTS file/env
    /// per-key, so the Info must stay EMPTY unless the user explicitly
    /// asked via CLI. The old code always set collective_buffering=true,
    /// which silently overrode romio_cb_* values injected by an external
    /// tuning harness -- the measured config was then not the candidate's.
    ///
    /// External hint channel (0.2.4): PIOB_ROMIO_HINTS="cb_nodes=4;
    /// romio_cb_write=enable" (pairs separated by ';' or ':', key=value)
    /// is injected FIRST, so explicit CLI flags still win per-key. The
    /// app-side MPI_Info at MPI_File_open is THE portable hint surface --
    /// it works on any MPI/ROMIO build, unlike env conventions (the
    /// ROMIO_HINTS file was removed in ROMIO 3.2; MPI_Info_env is an
    /// MPICH-family extension whose behaviour under OMPI's romio341 is
    /// unresolved). Values are passed through unvalidated: the tuning
    /// harness owns the key whitelist.
    Info make_romio_info() const {
        Info info;
        if (const char* env = std::getenv("PIOB_ROMIO_HINTS")) {
            std::string s(env);
            std::size_t pos = 0;
            while (pos < s.size()) {
                std::size_t end = s.find_first_of(";:", pos);
                if (end == std::string::npos) end = s.size();
                std::string kv = s.substr(pos, end - pos);
                const std::size_t eq = kv.find('=');
                if (eq != std::string::npos && eq + 1 < kv.size())
                    info.set(kv.substr(0, eq), kv.substr(eq + 1));
                pos = end + 1;
            }
        }
        if (!collective_buffering)
            info.set("collective_buffering", "false");  // explicit opt-out
        if (buffer_bytes)
            info.set("cb_buffer_size", std::to_string(*buffer_bytes));
        if (aggregators && *aggregators > 0)
            info.set("cb_nodes", std::to_string(*aggregators));
        return info;
    }
};

/// Wall time of the two payload phases, measured inside the backend
/// (rank-local view of the collective operation).
struct WriteSplit {
    double field_seconds{0};
    double particle_seconds{0};
};

template <typename B>
concept IoBackend = requires(B b, MPI_Comm comm, const IoSettings& s,
                             const std::filesystem::path& path, const Domain& d,
                             const Field3d<double>& f, const ParticleSet& ps,
                             Field3d<double>& dst, std::vector<Particle>& pdst) {
    { B::name } -> std::convertible_to<std::string_view>;
    { B::extension } -> std::convertible_to<std::string_view>;
    { B::available } -> std::convertible_to<bool>;
    B(comm, s);
    { b.write(path, d, f, ps) } -> std::same_as<WriteSplit>;
    { b.read_field(path, d, dst) } -> std::same_as<void>;
    // read a rank's own particle block back out of the checkpoint: the
    // verify path must cover EVERYTHING the write path wrote.
    { b.read_particles(path, d, ps, pdst) } -> std::same_as<void>;
};

} // namespace pio

// Concrete backends are pulled in (and static_assert'd) by pio/benchmark.hpp
// so that this header stays a dependency-free contract for the backends.
