// pio/backend_hdf5.hpp -- Parallel HDF5 backend.
//
// One file per checkpoint:
//   /fields/data    3-D double  (collective hyperslab, ghost-aware memspace)
//   /particles/{x,y,z,px,py,pz,mass}  1-D double, merged per rank
//   /particles/id   1-D uint64
//
// Particle writes have two strategies producing BYTE-IDENTICAL files:
//   * prepack (default): gather each AoS member into a contiguous staging
//     vector, then a plain contiguous H5Dwrite -- HDF5's fast path. Mirrors
//     what the adios2 backend does, so the backends compete on I/O, not on
//     who thought to pack first ("fair").
//   * strided (--h5-strided): hand HDF5 the AoS stride pattern (stride = 8
//     doubles) and let its generic selection machinery walk it element by
//     element. Kept to MEASURE the conversion cost, not to use.
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

        // ---- /particles/* : columnar member datasets -------------------------
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

        // member table: dataset name <-> memory location, double members 0..6
        struct Member {
            const char* name;
            std::size_t aos_off;                // offsetof into Particle
            double Particle::* dmem;            // typed access (prepack gather)
        };
        static constexpr Member members[7] = {
            {"particles/x", offsetof(Particle, x), &Particle::x},
            {"particles/y", offsetof(Particle, y), &Particle::y},
            {"particles/z", offsetof(Particle, z), &Particle::z},
            {"particles/px", offsetof(Particle, px), &Particle::px},
            {"particles/py", offsetof(Particle, py), &Particle::py},
            {"particles/pz", offsetof(Particle, pz), &Particle::pz},
            {"particles/mass", offsetof(Particle, mass), &Particle::mass}};

        // staging for prepack (inside the particle timing region on purpose:
        // the packing IS the cost the strided path hides inside HDF5)
        std::array<std::vector<double>, 7> stage_d;
        std::vector<std::uint64_t> stage_i;
        if (settings_.h5_prepack) {
            const std::size_t n = parts.local_count();
            for (auto& v : stage_d)
                v.resize(n);
            stage_i.resize(n);
            const Particle* p = parts.data();
            for (std::size_t i = 0; i < n; ++i) {
                for (int m = 0; m < 7; ++m)
                    stage_d[m][i] = p[i].*members[m].dmem;
                stage_i[i] = p[i].id;
            }
        }

        auto write_member = [&](const char* dset_name, hid_t h5type,
                                const void* mem_ptr, bool contiguous_mem) {
            detail::H5SpaceObj filespace{H5Screate_simple(1, &pgcount, nullptr)};
            detail::H5DsetObj dset{H5Dcreate(*file, dset_name, h5type, *filespace,
                                             *lcpl, H5P_DEFAULT, H5P_DEFAULT)};
            detail::H5SpaceObj fsub{H5Dget_space(*dset)};
            detail::h5_check(H5Sselect_hyperslab(*fsub, H5S_SELECT_SET, &pstart,
                                                 nullptr, &pcount, nullptr),
                             "particle filespace hyperslab");
            detail::H5SpaceObj memspace{
                H5Screate_simple(1, contiguous_mem ? &pcount : &pmem_extent,
                                 nullptr)};
            if (!contiguous_mem) {
                // memory: stride over the AoS records (see pmem_extent comment)
                const hsize_t mstart0 = 0;
                detail::h5_check(H5Sselect_hyperslab(*memspace, H5S_SELECT_SET,
                                                     &mstart0, &stride, &pcount,
                                                     &one),
                                 "particle memspace stride");
            }
            detail::h5_check(H5Dwrite(*dset, h5type, *memspace, *fsub, *xfer,
                                      mem_ptr),
                             "H5Dwrite particles");
        };

        const char* aos_base = reinterpret_cast<const char*>(parts.data());
        for (const auto& m : members)
            write_member(m.name, H5T_NATIVE_DOUBLE,
                         settings_.h5_prepack
                             ? static_cast<const void*>(stage_d[&m - members].data())
                             : aos_base + m.aos_off,
                         settings_.h5_prepack);
        write_member("particles/id", H5T_NATIVE_UINT64,
                     settings_.h5_prepack
                         ? static_cast<const void*>(stage_i.data())
                         : aos_base + offsetof(Particle, id),
                     settings_.h5_prepack);

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

    /// Read this rank's own particle block back: contiguous file range into
    /// contiguous temp buffers (HDF5 fast path), then assemble the AoS.
    void read_particles(const std::filesystem::path& path, const Domain&,
                        const ParticleSet& src, std::vector<Particle>& out) const {
        const std::size_t n = src.local_count();
        out.assign(n, Particle{});
        if (n == 0)
            return;

        Info info = settings_.make_romio_info();
        detail::H5PropObj fapl{H5Pcreate(H5P_FILE_ACCESS)};
        detail::h5_check(H5Pset_fapl_mpio(*fapl, comm_, info.get()), "H5Pset_fapl_mpio");
        detail::H5FileObj file{H5Fopen(path.c_str(), H5F_ACC_RDONLY, *fapl)};
        if (*file < 0)
            throw std::runtime_error("H5Fopen failed: " + path.string());
        detail::H5PropObj xfer{H5Pcreate(H5P_DATASET_XFER)};
        if (settings_.collective_buffering) {
            detail::h5_check(H5Pset_dxpl_mpio(*xfer, H5FD_MPIO_COLLECTIVE), "dxpl_mpio");
            detail::h5_check(H5Pset_dxpl_mpio_collective_opt(*xfer,
                                                             H5FD_MPIO_COLLECTIVE_IO),
                             "dxpl_mpio_collective_opt");
        }

        const hsize_t pstart = hsize_t(src.rank_offset());
        const hsize_t pcount = hsize_t(n);
        auto read_member = [&](const char* name, hid_t type, void* buf) {
            detail::H5DsetObj dset{H5Dopen(*file, name, H5P_DEFAULT)};
            if (*dset < 0)
                throw std::runtime_error(std::string("H5Dopen failed: ") + name);
            detail::H5SpaceObj fsub{H5Dget_space(*dset)};
            detail::h5_check(H5Sselect_hyperslab(*fsub, H5S_SELECT_SET, &pstart,
                                                 nullptr, &pcount, nullptr),
                             "particle read filespace");
            detail::H5SpaceObj memspace{H5Screate_simple(1, &pcount, nullptr)};
            detail::h5_check(H5Dread(*dset, type, *memspace, *fsub, *xfer, buf),
                             "H5Dread particles");
        };

        std::vector<double> cd(n);
        double Particle::*dmem[7] = {&Particle::x, &Particle::y, &Particle::z,
                                     &Particle::px, &Particle::py,
                                     &Particle::pz, &Particle::mass};
        const char* names[7] = {"particles/x", "particles/y", "particles/z",
                                "particles/px", "particles/py", "particles/pz",
                                "particles/mass"};
        for (int m = 0; m < 7; ++m) {
            read_member(names[m], H5T_NATIVE_DOUBLE, cd.data());
            for (std::size_t i = 0; i < n; ++i)
                out[i].*dmem[m] = cd[i];
        }
        std::vector<std::uint64_t> ci(n);
        read_member("particles/id", H5T_NATIVE_UINT64, ci.data());
        for (std::size_t i = 0; i < n; ++i)
            out[i].id = ci[i];
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
    void read_particles(const std::filesystem::path&, const Domain&,
                        const ParticleSet&, std::vector<Particle>&) const {
        throw std::logic_error("pio_bench built without HDF5 support");
    }
};

#endif

} // namespace pio
