// pio/domain.hpp -- 3-D Cartesian domain decomposition.
//
// One authoritative convention (the old code mixed x/y/z and z/y/x order
// between ci_pic and ci_pic_chkpt):
//   * logical/global order is always (x, y, z), C-order: x is slowest.
//   * `local`  : interior cells per rank
//   * `padded` : local + 2*ghost (memory layout of Field3d)
//   * `global` : dims * local      (file layout -- ghosts are NOT written)
//   * `start`  : coords * local    (this rank's interior origin in the file)
#pragma once

#include <array>
#include <utility>

#include "pio/mpi.hpp"

namespace pio {

struct Domain {
    static constexpr int ndim = 3;

    MPI_Comm cart{MPI_COMM_NULL};
    int rank{0};
    int size{1};
    std::array<int, 3> dims{};   // ranks per axis
    std::array<int, 3> coords{}; // this rank's position in the cartesian grid
    std::array<int, 3> local{};  // interior cells per rank
    std::array<int, 3> padded{}; // local + 2*ghost
    std::array<int, 3> global{}; // dims * local
    std::array<int, 3> start{};  // interior origin in the global grid
    int ghost{0};

    Domain() = default;

    static Domain create(MPI_Comm parent, const std::array<int, 3>& local, int ghost) {
        Domain d;
        d.local = local;
        d.ghost = ghost;
        d.size = size_of(parent);
        d.rank = rank_of(parent);

        int dims_c[ndim] = {0, 0, 0};
        PIO_MPI(MPI_Dims_create(d.size, ndim, dims_c));
        int periods_c[ndim] = {0, 0, 0};
        MPI_Comm cart = MPI_COMM_NULL;
        PIO_MPI(MPI_Cart_create(parent, ndim, dims_c, periods_c, /*reorder=*/0, &cart));
        d.cart = cart; // MPI_COMM_WORLD when topology is degenerate; still valid

        int coords_c[ndim];
        PIO_MPI(MPI_Cart_get(cart, ndim, dims_c, periods_c, coords_c));
        for (int i = 0; i < ndim; ++i) {
            d.dims[i] = dims_c[i];
            d.coords[i] = coords_c[i];
            d.padded[i] = local[i] + 2 * ghost;
            d.global[i] = dims_c[i] * local[i];
            d.start[i] = coords_c[i] * local[i];
        }
        return d;
    }

    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;
    Domain(Domain&& o) noexcept { *this = std::move(o); }
    Domain& operator=(Domain&& o) noexcept {
        if (this != &o) {
            free_cart();
            cart = o.cart;
            rank = o.rank;
            size = o.size;
            dims = o.dims;
            coords = o.coords;
            local = o.local;
            padded = o.padded;
            global = o.global;
            start = o.start;
            ghost = o.ghost;
            o.cart = MPI_COMM_NULL;
        }
        return *this;
    }
    ~Domain() { free_cart(); }

    std::size_t interior_count() const {
        return static_cast<std::size_t>(local[0]) * local[1] * local[2];
    }
    std::size_t padded_count() const {
        return static_cast<std::size_t>(padded[0]) * padded[1] * padded[2];
    }

private:
    void free_cart() {
        if (cart != MPI_COMM_NULL && cart != MPI_COMM_WORLD)
            MPI_Comm_free(&cart);
        cart = MPI_COMM_NULL;
    }
};

} // namespace pio
