// pio/backend_hdf5.hpp -- Parallel HDF5 backend.
//
// One file per checkpoint:
//   /fields/data    3-D double  (collective hyperslab, ghost-aware memspace)
//   /particles/{x,y,z,px,py,pz,mass}  1-D double, merged per rank
//   /particles/id   1-D uint64
// Particle datasets are written directly out of the AoS buffer using a
// strided memory dataspace (stride = 8 doubles) -- no SoA temporaries.
//
// NOTE (bug fixed vs old ci_pic_chkpt::p_write_hdf5): the old code passed
// a simple local-dims memspace together with an &arr[g][g][g] pointer,
// which silently wrote wrong data whenever ghost > 0 (interior rows are
// NOT contiguous in a padded array). Here the memspace is the full padded
// extent with an interior hyperslab selection -- correct for any ghost.
#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "pio/backends.hpp"

#ifdef PIO_HAVE_HDF5
#include <hdf5.h>
#ifndef H5_HAVE_PARALLEL
#error "PIO_HAVE_HDF5 requires a PARALLEL HDF5 build (H5Pset_fapl_mpio missing in serial builds)"
#endif
#endif

namespace pio {

#ifdef PIO_HAVE_HDF5

namespace detail {

inline void h5_check(herr_t rc, const char* what) {
    if (rc < 0)
        throw std::runtime_error(std::string("HDF5 failure: ") + what);
}

/// RAII for hid_t families (all closed by their *close(hid_t)).
template <auto (*CloseFn)(hid_t)>
class H5Obj {
public:
    H5Obj() = default;
    explicit H5Obj(hid_t id) : id_(id) {}
    H5Obj(const H5Obj&) = delete;
    H5Obj& operator=(const H5Obj&) = delete;
    H5Obj(H5Obj&& o) noexcept : id_(std::exchange(o.id_, -1)) {}
    H5Obj& operator=(H5Obj&& o) noexcept {
        if (this != &o) {
            close();
            id_ = std::exchange(o.id_, -1);
        }
        return *this;
    }
    ~H5Obj() { close(); }
    hid_t operator*() const { return id_; }
    void close() {
        if (id_ >= 0) {
            CloseFn(id_);
            id_ = -1;
        }
    }

private:
    hid_t id_{-1};
};

using H5FileObj = H5Obj<&H5Fclose>;
using H5DsetObj = H5Obj<&H5Dclose>;
using H5SpaceObj = H5Obj<&H5Sclose>;
using H5PropObj = H5Obj<&H5Pclose>;

} // namespace detail

class Hdf5Backend {
public:
    static constexpr std::string_view name = "hdf5";
    static constexpr std::string_view extension = ".h5";
    static constexpr bool available = true;

    explicit Hdf5Backend(MPI_Comm comm, const IoSettings& s)
        : comm_(comm), settings_(s) {}

    WriteSplit write(const std::filesystem::path& path, const Domain& d,
                     const Field3d<double>& field, const ParticleSet& parts) const {
        WriteSplit split;
        Info info = settings_.make_romio_info();

        detail::H5PropObj fapl{H5Pcreate(H5P_FILE_ACCESS)};
        detail::h5_check(H5Pset_fapl_mpio(*fapl, comm_, info.get()), "H5Pset_fapl_mpio");
        detail::H5FileObj file{H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, *fapl)};
        if (*file < 0)
            throw std::runtime_error("H5Fcreate failed: " + path.string());

        // ---- /fields/data ----------------------------------------------------
        const hsize_t gdims[3] = {hsize_t(d.global[0]), hsize_t(d.global[1]), hsize_t(d.global[2])};
        const hsize_t start[3] = {hsize_t(d.start[0]), hsize_t(d.start[1]), hsize_t(d.start[2])};
        const hsize_t count[3] = {hsize_t(d.local[0]), hsize_t(d.local[1]), hsize_t(d.local[2])};
        const hsize_t pdims[3] = {hsize_t(d.padded[0]), hsize_t(d.padded[1]), hsize_t(d.padded[2])};
        const hsize_t mstart[3] = {hsize_t(d.ghost), hsize_t(d.ghost), hsize_t(d.ghost)};

        detail::H5PropObj xfer{H5Pcreate(H5P_DATASET_XFER)};
        if (settings_.collective_buffering) {
            detail::h5_check(H5Pset_dxpl_mpio(*xfer, H5FD_MPIO_COLLECTIVE), "dxpl_mpio");
            // HDF5's "collective" transfer mode by default still issues
            // INDEPENDENT MPI-IO calls internally (coll_write=independent):
            // the strided AoS-member selections below then get NO ROMIO
            // aggregation -- the classic slow strided path. Request
            // collective MPI-IO calls so ROMIO aggregates the gathers.
            // (Only valid when the transfer mode above is COLLECTIVE.)
            detail::h5_check(H5Pset_dxpl_mpio_collective_opt(*xfer,
                                                             H5FD_MPIO_COLLECTIVE_IO),
                             "dxpl_mpio_collective_opt");
        }

        // "/fields/data" and "particles/*" live in groups: the default
        // LCPL does NOT create intermediate groups (H5L "component not
        // found") -- request automatic creation once, reuse everywhere.
        detail::H5PropObj lcpl{H5Pcreate(H5P_LINK_CREATE)};
        detail::h5_check(H5Pset_create_intermediate_group(*lcpl, 1),
                         "link create intermediate group");

        Stopwatch w;
        {
            detail::H5SpaceObj filespace{H5Screate_simple(3, gdims, nullptr)};
            detail::H5DsetObj dset{H5Dcreate(*file, "/fields/data", H5T_NATIVE_DOUBLE,
                                             *filespace, *lcpl, H5P_DEFAULT,
                                             H5P_DEFAULT)};
            detail::H5SpaceObj fsub{H5Dget_space(*dset)};
            detail::h5_check(H5Sselect_hyperslab(*fsub, H5S_SELECT_SET, start, nullptr,
                                                 count, nullptr),
                             "field filespace hyperslab");
            detail::H5SpaceObj memspace{H5Screate_simple(3, pdims, nullptr)};
            detail::h5_check(H5Sselect_hyperslab(*memspace, H5S_SELECT_SET, mstart,
                                                 nullptr, count, nullptr),
                             "field memspace hyperslab");
            detail::h5_check(H5Dwrite(*dset, H5T_NATIVE_DOUBLE, *memspace, *fsub,
                                      *xfer, field.data()),
                             "H5Dwrite fields");
        }
        split.field_seconds = w.elapsed();

        // ---- /particles/* : strided reads straight out of the AoS ----------
        w.restart();
        const hsize_t pgcount = hsize_t(parts.global_count());
        const hsize_t pstart = hsize_t(parts.rank_offset());
        const hsize_t pcount = hsize_t(parts.local_count());
        const hsize_t one = 1;
        const hsize_t stride = sizeof(Particle) / sizeof(double); // 8
        // The strided selection must FIT the memory dataspace: use the
        // standard AoS trick -- a virtual extent of pcount*stride elements
        // (the full record span) and stride through it picking members.
        const hsize_t pmem_extent = pcount * stride;

        auto write_member = [&](const char* dset_name, std::size_t member_offset,
                                hid_t h5type) {
            detail::H5SpaceObj filespace{H5Screate_simple(1, &pgcount, nullptr)};
            detail::H5DsetObj dset{H5Dcreate(*file, dset_name, h5type, *filespace,
                                             *lcpl, H5P_DEFAULT, H5P_DEFAULT)};
            detail::H5SpaceObj fsub{H5Dget_space(*dset)};
            detail::h5_check(H5Sselect_hyperslab(*fsub, H5S_SELECT_SET, &pstart,
                                                 nullptr, &pcount, nullptr),
                             "particle filespace hyperslab");
            // memory: stride over the AoS records (see pmem_extent comment)
            detail::H5SpaceObj memspace{H5Screate_simple(1, &pmem_extent, nullptr)};
            const hsize_t mstart0 = 0;
            detail::h5_check(H5Sselect_hyperslab(*memspace, H5S_SELECT_SET, &mstart0,
                                                 &stride, &pcount, &one),
                             "particle memspace stride");
            const void* base = static_cast<const char*>(
                static_cast<const void*>(parts.data())) + member_offset;
            detail::h5_check(H5Dwrite(*dset, h5type, *memspace, *fsub, *xfer, base),
                             "H5Dwrite particles");
        };

        for (auto [nm, off] : {std::pair{"particles/x", offsetof(Particle, x)},
                               {"particles/y", offsetof(Particle, y)},
                               {"particles/z", offsetof(Particle, z)},
                               {"particles/px", offsetof(Particle, px)},
                               {"particles/py", offsetof(Particle, py)},
                               {"particles/pz", offsetof(Particle, pz)},
                               {"particles/mass", offsetof(Particle, mass)}})
            write_member(nm, off, H5T_NATIVE_DOUBLE);
        write_member("particles/id", offsetof(Particle, id), H5T_NATIVE_UINT64);

        split.particle_seconds = w.elapsed();
        file.close(); // collective close point
        return split;
    }

    void read_field(const std::filesystem::path& path, const Domain& d,
                    Field3d<double>& dst) const {
        Info info = settings_.make_romio_info();
        detail::H5PropObj fapl{H5Pcreate(H5P_FILE_ACCESS)};
        detail::h5_check(H5Pset_fapl_mpio(*fapl, comm_, info.get()), "H5Pset_fapl_mpio");
        detail::H5FileObj file{H5Fopen(path.c_str(), H5F_ACC_RDONLY, *fapl)};
        detail::H5DsetObj dset{H5Dopen(*file, "/fields/data", H5P_DEFAULT)};

        const hsize_t start[3] = {hsize_t(d.start[0]), hsize_t(d.start[1]), hsize_t(d.start[2])};
        const hsize_t count[3] = {hsize_t(d.local[0]), hsize_t(d.local[1]), hsize_t(d.local[2])};
        const hsize_t pdims[3] = {hsize_t(d.padded[0]), hsize_t(d.padded[1]), hsize_t(d.padded[2])};
        const hsize_t mstart[3] = {hsize_t(d.ghost), hsize_t(d.ghost), hsize_t(d.ghost)};

        detail::H5SpaceObj fsub{H5Dget_space(*dset)};
        detail::h5_check(H5Sselect_hyperslab(*fsub, H5S_SELECT_SET, start, nullptr,
                                             count, nullptr),
                         "read filespace hyperslab");
        detail::H5SpaceObj memspace{H5Screate_simple(3, pdims, nullptr)};
        detail::h5_check(H5Sselect_hyperslab(*memspace, H5S_SELECT_SET, mstart, nullptr,
                                             count, nullptr),
                         "read memspace hyperslab");
        detail::H5PropObj xfer{H5Pcreate(H5P_DATASET_XFER)};
        if (settings_.collective_buffering) {
            detail::h5_check(H5Pset_dxpl_mpio(*xfer, H5FD_MPIO_COLLECTIVE), "dxpl_mpio");
            detail::h5_check(H5Pset_dxpl_mpio_collective_opt(*xfer,
                                                             H5FD_MPIO_COLLECTIVE_IO),
                             "dxpl_mpio_collective_opt");
        }
        detail::h5_check(H5Dread(*dset, H5T_NATIVE_DOUBLE, *memspace, *fsub, *xfer,
                                 dst.data()),
                         "H5Dread fields");
    }

private:
    MPI_Comm comm_;
    IoSettings settings_;
};

#else // !PIO_HAVE_HDF5 -- compile-time stub keeps the concept satisfied

class Hdf5Backend {
public:
    static constexpr std::string_view name = "hdf5";
    static constexpr std::string_view extension = ".h5";
    static constexpr bool available = false;

    explicit Hdf5Backend(MPI_Comm, const IoSettings&) {}
    WriteSplit write(const std::filesystem::path&, const Domain&,
                     const Field3d<double>&, const ParticleSet&) const {
        throw std::logic_error("pio_bench built without HDF5 support");
    }
    void read_field(const std::filesystem::path&, const Domain&, Field3d<double>&) const {
        throw std::logic_error("pio_bench built without HDF5 support");
    }
};

#endif

} // namespace pio
