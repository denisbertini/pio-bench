// pio/field.hpp -- contiguous 3-D field with ghost cells.
//
// Replaces `double*** m_data` (triple pointer, three allocation failure
// points, cache-hostile rows scattered in the heap). One std::vector with
// padded extents: ghost-aware subarray/hyperslab views then reference the
// interior block directly -- no manual &arr[g][g][g] arithmetic at call
// sites (which, combined with a wrong memspace, was the ghost-cell bug in
// the old ci_pic_chkpt HDF5 writer).
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "pio/domain.hpp"

namespace pio {

template <typename T = double>
class Field3d {
public:
    Field3d() = default;

    /// Allocates padded extents (local + 2*ghost), zero-filled ghosts.
    explicit Field3d(const Domain& d)
        : pdim_(d.padded), ghost_(d.ghost), buf_(d.padded_count(), T{0}) {}

    /// padded coordinates (0 .. padded-1 on each axis)
    T& at(int i, int j, int k) {
        return buf_[(static_cast<std::size_t>(i) * pdim_[1] + j) * pdim_[2] + k];
    }
    const T& at(int i, int j, int k) const {
        return buf_[(static_cast<std::size_t>(i) * pdim_[1] + j) * pdim_[2] + k];
    }

    T* data() { return buf_.data(); }
    const T* data() const { return buf_.data(); }

    /// Pointer to interior cell (ghost, ghost, ghost). Used by backends
    /// whose memory view is the plain interior block (ADIOS2).
    T* interior_ptr() {
        return buf_.data() +
               (static_cast<std::size_t>(ghost_) * pdim_[1] + ghost_) * pdim_[2] + ghost_;
    }
    const T* interior_ptr() const {
        return buf_.data() +
               (static_cast<std::size_t>(ghost_) * pdim_[1] + ghost_) * pdim_[2] + ghost_;
    }

    const std::array<int, 3>& padded() const { return pdim_; }
    int ghost() const { return ghost_; }
    std::size_t padded_count() const { return buf_.size(); }
    std::size_t interior_count() const {
        std::size_t n = 1;
        for (int i = 0; i < 3; ++i)
            n *= static_cast<std::size_t>(pdim_[i] - 2 * ghost_);
        return n;
    }

    /// Deterministic, position-unique pattern over the interior (ghosts
    /// stay zero): value = flattened global C-order index, exactly
    /// representable as double for grids < 2^53 cells.
    void fill_pattern(const Domain& d) {
        const auto [lx, ly, lz] = d.global;
        for (int i = 0; i < local(d)[0]; ++i)
            for (int j = 0; j < local(d)[1]; ++j)
                for (int k = 0; k < local(d)[2]; ++k) {
                    const int gx = d.start[0] + i, gy = d.start[1] + j, gz = d.start[2] + k;
                    const auto idx = static_cast<double>(
                        (static_cast<std::size_t>(gx) * ly + gy) * lz + gz);
                    at(ghost_ + i, ghost_ + j, ghost_ + k) = static_cast<T>(idx);
                }
    }

private:
    static const std::array<int, 3>& local(const Domain& d) { return d.local; }

    std::array<int, 3> pdim_{};
    int ghost_{0};
    std::vector<T> buf_;
};

} // namespace pio
