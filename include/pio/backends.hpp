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
#include <filesystem>
#include <optional>
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

    static IoSettings from(const BenchConfig& c) {
        return IoSettings{c.aggregators, c.buffer_bytes, c.collective_buffering,
                          c.h5_prepack};
    }

    /// ROMIO hint set shared by the mpiio and hdf5 (H5Pset_fapl_mpio) paths.
    Info make_romio_info() const {
        Info info;
        info.set_if(collective_buffering, "collective_buffering", "true");
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
