// pio/backend_mpiio.hpp -- raw MPI-IO backend (single shared .bin file:
// field block at displacement 0, particle records after the field bytes).
#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "pio/backends.hpp"

namespace pio {

class MpiIoBackend {
public:
    static constexpr std::string_view name = "mpiio";
    static constexpr std::string_view extension = ".bin";
    static constexpr bool available = true;

    explicit MpiIoBackend(MPI_Comm comm, const IoSettings& s)
        : comm_(comm), settings_(s) {}

    WriteSplit write(const std::filesystem::path& path, const Domain& d,
                     const Field3d<double>& field, const ParticleSet& parts) const {
        Info info = settings_.make_romio_info();
        WriteSplit split;

        // ---- fields: global subarray view, ghost-aware memory type ----
        const MPI_Aint gs[3] = {d.global[0], d.global[1], d.global[2]};
        const MPI_Aint ss[3] = {d.local[0], d.local[1], d.local[2]};
        const MPI_Aint st[3] = {d.start[0], d.start[1], d.start[2]};
        const MPI_Aint ms[3] = {d.padded[0], d.padded[1], d.padded[2]};
        const MPI_Aint mst[3] = {d.ghost, d.ghost, d.ghost};

        Datatype file_type = Datatype::subarray(3, gs, ss, st, MPI_DOUBLE);
        Datatype mem_type = Datatype::subarray(3, ms, ss, mst, MPI_DOUBLE);

        File fh = File::open(comm_, path.string(), MPI_MODE_CREATE | MPI_MODE_RDWR,
                             info.get());
        fh.set_view(0, MPI_DOUBLE, file_type, info.get());

        Stopwatch w;
        PIO_MPI(MPI_File_write_all(fh.get(), field.data(), 1, mem_type, MPI_STATUS_IGNORE));
        split.field_seconds = w.elapsed();

        // ---- particles: 1-D global record array, appended after fields ----
        const MPI_Aint lcount = static_cast<MPI_Aint>(parts.local_count());
        if (lcount > std::numeric_limits<int>::max())
            throw MpiError("particle block exceeds int count limit");
        const MPI_Aint gcount = static_cast<MPI_Aint>(parts.global_count());
        const MPI_Offset disp =
            static_cast<MPI_Offset>(gs[0]) * gs[1] * gs[2] * sizeof(double);

        const MPI_Datatype rec = ParticleSet::record_type();
        Datatype pfile = Datatype::subarray_1d(gcount, lcount,
                                               static_cast<MPI_Aint>(parts.rank_offset()), rec);
        Datatype pmem = Datatype::contiguous(lcount, rec);

        w.restart();
        fh.set_view(disp, rec, pfile, info.get());
        PIO_MPI(MPI_File_write_all(fh.get(), const_cast<Particle*>(parts.data()), 1,
                                   pmem, MPI_STATUS_IGNORE));
        split.particle_seconds = w.elapsed();

        fh.close(); // deterministic collective close point
        return split;
    }

    void read_field(const std::filesystem::path& path, const Domain& d,
                    Field3d<double>& dst) const {
        const MPI_Aint gs[3] = {d.global[0], d.global[1], d.global[2]};
        const MPI_Aint ss[3] = {d.local[0], d.local[1], d.local[2]};
        const MPI_Aint st[3] = {d.start[0], d.start[1], d.start[2]};
        const MPI_Aint ms[3] = {d.padded[0], d.padded[1], d.padded[2]};
        const MPI_Aint mst[3] = {d.ghost, d.ghost, d.ghost};

        Datatype file_type = Datatype::subarray(3, gs, ss, st, MPI_DOUBLE);
        Datatype mem_type = Datatype::subarray(3, ms, ss, mst, MPI_DOUBLE);

        File fh = File::open(comm_, path.string(), MPI_MODE_RDONLY, MPI_INFO_NULL);
        fh.set_view(0, MPI_DOUBLE, file_type, MPI_INFO_NULL);
        PIO_MPI(MPI_File_read_all(fh.get(), dst.data(), 1, mem_type, MPI_STATUS_IGNORE));
        fh.close();
    }

    /// Read this rank's own record block back from after the field bytes
    /// (default byte view + absolute collective read -- no subarray needed).
    void read_particles(const std::filesystem::path& path, const Domain& d,
                        const ParticleSet& src, std::vector<Particle>& out) const {
        const std::size_t n = src.local_count();
        out.assign(n, Particle{});
        if (static_cast<MPI_Aint>(n) > std::numeric_limits<int>::max())
            throw MpiError("particle block exceeds int count limit");

        const MPI_Offset disp =
            static_cast<MPI_Offset>(d.global[0]) * d.global[1] * d.global[2] *
                sizeof(double) +
            static_cast<MPI_Offset>(src.rank_offset()) * sizeof(Particle);

        File fh = File::open(comm_, path.string(), MPI_MODE_RDONLY, MPI_INFO_NULL);
        PIO_MPI(MPI_File_read_at_all(fh.get(), disp, out.data(),
                                     static_cast<int>(n),
                                     ParticleSet::record_type(), MPI_STATUS_IGNORE));
        fh.close();
    }

private:
    MPI_Comm comm_;
    IoSettings settings_;
};

} // namespace pio
