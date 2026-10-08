// pio/particles.hpp -- particle payload for the PIC-like benchmark.
//
// AoS layout (std::vector<Particle>) as requested. For the wire format:
//   * MPI-IO : one derived record datatype (7 double + uint64), the whole
//              per-rank block written into the SAME shared file as the
//              fields, at a displacement after the field bytes.
//   * HDF5   : per-member 1-D datasets; by default each member is pre-packed
//              into a contiguous staging vector (fast H5Dwrite path);
//              --h5-strided instead hands HDF5 the AoS stride pattern.
//              Both produce identical files.
//   * ADIOS2 : per-member 1-D variables with explicit global shape and
//              per-rank blocks (ConstantDims); the engine merges them
//              into the global arrays.
//
// Determinism: RNG seeded from (seed, rank) only, and id = global particle
// index -- so a reader can verify provenance of every particle cheaply.
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "pio/mpi.hpp"

namespace pio {

struct Particle {
    double x{0}, y{0}, z{0};
    double px{0}, py{0}, pz{0};
    double mass{1.0};
    std::uint64_t id{0};
};
static_assert(std::is_standard_layout_v<Particle>,
              "Particle must be standard-layout for MPI/HDF5 derived types");
static_assert(sizeof(Particle) == 8 * sizeof(double),
              "Particle must be a padding-free 64B record");

class ParticleSet {
public:
    ParticleSet() = default;

    /// n identical-sized blocks per rank (uniform is what checkpoint
    /// writers see most often and keeps the subarray math exact).
    /// Global offset per rank via MPI_Exscan, global count via Allreduce.
    static ParticleSet generate(MPI_Comm comm, std::size_t n, std::uint64_t seed) {
        ParticleSet ps;
        ps.p_.resize(n);

        const std::uint64_t local = n;
        std::uint64_t gcount = 0, offset = 0;
        PIO_MPI(MPI_Allreduce(&local, &gcount, 1, MPI_UINT64_T, MPI_SUM, comm));
        PIO_MPI(MPI_Exscan(&local, &offset, 1, MPI_UINT64_T, MPI_SUM, comm));
        if (rank_of(comm) == 0)
            offset = 0; // Exscan leaves rank 0 undefined
        ps.global_count_ = gcount;
        ps.rank_offset_ = offset;

        // Deterministic per (seed, rank): reproducible across runs AND
        // across rank counts only in the trivial case -- documented.
        std::seed_seq seq{seed, static_cast<std::uint64_t>(rank_of(comm))};
        std::mt19937_64 rng(seq);
        std::uniform_real_distribution<double> pos(0.0, 1.0);
        std::normal_distribution<double> mom(0.0, 1.0);

        for (std::size_t i = 0; i < n; ++i) {
            Particle& p = ps.p_[i];
            p.x = pos(rng);
            p.y = pos(rng);
            p.z = pos(rng);
            p.px = mom(rng);
            p.py = mom(rng);
            p.pz = mom(rng);
            p.mass = 1.0;
            p.id = offset + i; // globally unique, provenance-encoded
        }
        return ps;
    }

    std::size_t local_count() const { return p_.size(); }
    std::uint64_t global_count() const { return global_count_; }
    std::uint64_t rank_offset() const { return rank_offset_; }

    Particle* data() { return p_.data(); }
    const Particle* data() const { return p_.data(); }
    const std::vector<Particle>& particles() const { return p_; }

    /// Committed MPI record datatype for Particle. Process-lifetime cache
    /// (one per program): deliberately never freed, created on first use.
    static MPI_Datatype record_type() {
        static const MPI_Datatype type = [] {
            // standard-layout => offsetof is well-defined
            const std::array<MPI_Aint, 8> offsets{
                offsetof(Particle, x), offsetof(Particle, y),
                offsetof(Particle, z), offsetof(Particle, px),
                offsetof(Particle, py), offsetof(Particle, pz),
                offsetof(Particle, mass), offsetof(Particle, id)};
            std::array<MPI_Datatype, 8> types{MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                              MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                              MPI_DOUBLE, MPI_UINT64_T};
            std::array<int, 8> blens{1, 1, 1, 1, 1, 1, 1, 1};
            MPI_Datatype t = MPI_DATATYPE_NULL;
            PIO_MPI(MPI_Type_create_struct(8, blens.data(), offsets.data(),
                                           types.data(), &t));
            PIO_MPI(MPI_Type_commit(&t));
            return t;
        }();
        return type;
    }

private:
    std::vector<Particle> p_;
    std::uint64_t global_count_{0};
    std::uint64_t rank_offset_{0};
};

} // namespace pio
