// pio/backend_mpiio.hpp -- raw MPI-IO backend (single shared .bin file:
// field block at displacement 0, particle records after the field bytes).
//
// View discipline (one view change per checkpoint, was two):
//   * A freshly opened file has the DEFAULT view: etype MPI_BYTE,
//     filetype MPI_BYTE (MPI-4.0 9.6.1). The particle records are written
//     on that byte view with a COLLECTIVE _at_ call at each rank's own
//     byte displacement -- contiguous per-rank blocks need no set_view at
//     all. (MPI_File_set_view is a full-communicator collective: the old
//     code paid two of them per checkpoint, one for the field subarray and
//     a second to switch to a record-subarray particle view.)
//   * Then ONE set_view switches to the 3-D field subarray for the
//     collective write_all. Ghost handling stays exactly as before: the
//     memory type is a subarray over the padded extent.
//   * Counts use the 64-bit MPI-IO count API when the stack provides it
//     (probed in CMake; see pio/mpi.hpp file_write_at_all & co).
//
// Datatype cache: the subarray types depend only on the (constant-per-run)
// domain geometry, so they are created ONCE and reused by every
// checkpoint's write AND read instead of being re-created/re-committed on
// each call.
#pragma once

#include <array>
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

        File fh = File::open(comm_, path.string(), MPI_MODE_CREATE | MPI_MODE_RDWR,
                             info.get());

        // ---- particles: record blocks after the field bytes (byte view) ----
        // Per-rank displacement: <all field bytes> + this rank's record
        // offset. The op is collective; ranks may pass different
        // displacements (the read path already relied on that).
        const MPI_Aint lcount_aint = static_cast<MPI_Aint>(parts.local_count());
        if (lcount_aint > static_cast<MPI_Aint>(max_file_count()))
            throw MpiError("particle block exceeds the MPI-IO count limit of "
                           "this stack (needs an MPI-4.0 MPI_Count build)");
        const FileCount lcount = static_cast<FileCount>(lcount_aint);
        const MPI_Offset disp =
            static_cast<MPI_Offset>(d.global[0]) * d.global[1] * d.global[2] *
            sizeof(double) +
            static_cast<MPI_Offset>(parts.rank_offset()) * sizeof(Particle);

        Stopwatch w;
        PIO_MPI(file_write_at_all(fh.get(), disp, parts.data(), lcount,
                                  ParticleSet::record_type()));
        split.particle_seconds = w.elapsed();

        // ---- fields: the single view change, 3-D global subarray --------
        const FieldTypes& ft = field_types(d);
        w.restart();
        fh.set_view(0, MPI_DOUBLE, ft.file_type.get(), info.get());
        PIO_MPI(MPI_File_write_all(fh.get(), field.data(), 1, ft.mem_type.get(),
                                   MPI_STATUS_IGNORE));
        split.field_seconds = w.elapsed();

        fh.close(); // deterministic collective close point
        return split;
    }

    void read_field(const std::filesystem::path& path, const Domain& d,
                    Field3d<double>& dst) const {
        const FieldTypes& ft = field_types(d);

        File fh = File::open(comm_, path.string(), MPI_MODE_RDONLY, MPI_INFO_NULL);
        fh.set_view(0, MPI_DOUBLE, ft.file_type.get(), MPI_INFO_NULL);
        PIO_MPI(MPI_File_read_all(fh.get(), dst.data(), 1, ft.mem_type.get(),
                                  MPI_STATUS_IGNORE));
        fh.close();
    }

    /// Read this rank's own record block back from after the field bytes
    /// (default byte view + absolute collective read -- no subarray needed).
    void read_particles(const std::filesystem::path& path, const Domain& d,
                        const ParticleSet& src, std::vector<Particle>& out) const {
        const std::size_t n = src.local_count();
        out.assign(n, Particle{});
        const MPI_Aint n_aint = static_cast<MPI_Aint>(n);
        if (n_aint > static_cast<MPI_Aint>(max_file_count()))
            throw MpiError("particle block exceeds the MPI-IO count limit of "
                           "this stack (needs an MPI-4.0 MPI_Count build)");

        const MPI_Offset disp =
            static_cast<MPI_Offset>(d.global[0]) * d.global[1] * d.global[2] *
            sizeof(double) +
            static_cast<MPI_Offset>(src.rank_offset()) * sizeof(Particle);

        File fh = File::open(comm_, path.string(), MPI_MODE_RDONLY, MPI_INFO_NULL);
        PIO_MPI(file_read_at_all(fh.get(), disp, out.data(),
                                 static_cast<FileCount>(n_aint),
                                 ParticleSet::record_type()));
        fh.close();
    }

private:
    /// Per-run cache of the field subarray datatypes: they depend only on
    /// the domain geometry, which is constant for the whole run. Built on
    /// first use (write or read), validated on every subsequent call.
    struct FieldTypes {
        std::array<MPI_Aint, 3> global{}, local{}, start{}, padded{};
        int ghost{-1};
        bool valid{false};
        Datatype file_type, mem_type;
    };
    mutable FieldTypes field_types_{};

    const FieldTypes& field_types(const Domain& d) const {
        const std::array<MPI_Aint, 3> g{static_cast<MPI_Aint>(d.global[0]),
                                         static_cast<MPI_Aint>(d.global[1]),
                                         static_cast<MPI_Aint>(d.global[2])};
        const std::array<MPI_Aint, 3> l{static_cast<MPI_Aint>(d.local[0]),
                                         static_cast<MPI_Aint>(d.local[1]),
                                         static_cast<MPI_Aint>(d.local[2])};
        const std::array<MPI_Aint, 3> s{static_cast<MPI_Aint>(d.start[0]),
                                         static_cast<MPI_Aint>(d.start[1]),
                                         static_cast<MPI_Aint>(d.start[2])};
        const std::array<MPI_Aint, 3> p{static_cast<MPI_Aint>(d.padded[0]),
                                         static_cast<MPI_Aint>(d.padded[1]),
                                         static_cast<MPI_Aint>(d.padded[2])};
        if (field_types_.valid && field_types_.global == g &&
            field_types_.local == l && field_types_.start == s &&
            field_types_.padded == p && field_types_.ghost == d.ghost)
            return field_types_;

        const MPI_Aint mst[3] = {d.ghost, d.ghost, d.ghost};
        field_types_.file_type =
            Datatype::subarray(3, g.data(), l.data(), s.data(), MPI_DOUBLE);
        field_types_.mem_type =
            Datatype::subarray(3, p.data(), l.data(), mst, MPI_DOUBLE);
        field_types_.global = g;
        field_types_.local = l;
        field_types_.start = s;
        field_types_.padded = p;
        field_types_.ghost = d.ghost;
        field_types_.valid = true;
        return field_types_;
    }

    MPI_Comm comm_;
    IoSettings settings_;
};

} // namespace pio
