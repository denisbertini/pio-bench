// pio/benchmark.hpp -- the merged benchmark: ci_pic + ci_pic_chkpt, once,
// parameterized over an IoBackend policy.
//
// Structure inherited from the old classes:
//   * 3-D Cartesian decomposition, checkpoint-every-N-steps loop,
//     optional simulated compute, optional sampled read-back verification
//     (from ci_pic_chkpt); per-step write-rate sampling (from ci_pic).
// Modernized:
//   * backend choice is a template parameter (concept-checked), not an
//     enum + if-ladder replicated in 6 methods;
//   * fields AND particles in one payload, one file, one timing split;
//   * exceptions instead of print-and-continue; RAII everywhere;
//   * machine-readable metrics (JSON Lines) alongside human output.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "pio/backends.hpp"
#include "pio/backend_adios2.hpp"
#include "pio/backend_hdf5.hpp"
#include "pio/backend_mpiio.hpp"
#include "pio/domain.hpp"
#include "pio/field.hpp"
#include "pio/mpi.hpp"
#include "pio/params.hpp"
#include "pio/particles.hpp"
#include "pio/stats.hpp"

namespace pio {

/// Instrument version: stamped into every metrics line so fitness data
/// always self-documents the binary that produced it (bump on ANY change
/// that can move measured numbers; tag the repo in lockstep).
inline constexpr const char* kBenchmarkVersion = "0.1.0";

static_assert(IoBackend<MpiIoBackend>);
static_assert(IoBackend<Hdf5Backend>);
static_assert(IoBackend<Adios2Backend>);

template <IoBackend Backend>
class PicBenchmark {
public:
    PicBenchmark(const BenchConfig& cfg, MPI_Comm comm)
        : cfg_(cfg), comm_(comm), rank_(rank_of(comm)),
          domain_(Domain::create(comm, cfg.local, cfg.ghost)),
          field_(domain_),
          particles_(ParticleSet::generate(comm, cfg.particles_per_rank, cfg.seed)),
          backend_(comm, IoSettings::from(cfg)) {
        field_.fill_pattern(domain_);
        field_bytes_ = domain_.interior_count() * sizeof(double);
        particle_bytes_ = particles_.local_count() * sizeof(Particle);
    }

    int run() {
        log_banner();

        RunningStats total_stats, field_stats, part_stats;
        int chkpt_id = 0;

        for (int step = 1; step <= cfg_.steps; ++step) {
            if (cfg_.compute_ms > 0.0)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(static_cast<long long>(cfg_.compute_ms)));

            const bool due = (step % cfg_.chkpt_interval == 0) || (step == cfg_.steps);
            if (!due || (cfg_.max_checkpoints >= 0 && chkpt_id >= cfg_.max_checkpoints))
                continue;
            ++chkpt_id;

            const auto path = chkpt_path(chkpt_id, step);
            PIO_MPI(MPI_Barrier(comm_)); // exclude loop bookkeeping
            const Stopwatch wall;
            const WriteSplit split = backend_.write(path, domain_, field_, particles_);
            PIO_MPI(MPI_Barrier(comm_)); // completion point for all ranks
            const double total_s = wall.elapsed();

            const std::uint64_t bytes = field_bytes_ + particle_bytes_;
            const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);

            bool verified = true;
            if (cfg_.verify)
                verified = verify_checkpoint(path);

            if (rank_ == 0) {
                total_stats.add(total_s);
                field_stats.add(split.field_seconds);
                part_stats.add(split.particle_seconds);
                if (!cfg_.quiet)
                    std::cout << "chkpt " << chkpt_id << " (step " << step << "): "
                              << std::fixed << std::setprecision(3) << total_s << " s, "
                              << mib / total_s << " MiB/s  [field "
                              << split.field_seconds << " s | particles "
                              << split.particle_seconds << " s]"
                              << (cfg_.verify ? (verified ? " | verified" : " | MISMATCH")
                                              : "")
                              << '\n';
                append_metric(chkpt_id, step, path, total_s, split, bytes);
            }

            if (!verified) {
                // Fail loudly -- a benchmark that silently corrupts is worse
                // than none (the old code just counted failures).
                if (rank_ == 0)
                    std::cerr << "-E- : verification failed for " << path << "\n";
                MPI_Abort(comm_, 1);
            }

            if (!cfg_.keep_files) {
                PIO_MPI(MPI_Barrier(comm_)); // all ranks done with the file
                if (rank_ == 0) {
                    std::error_code ec;
                    std::filesystem::remove_all(path, ec); // dirs (.bp) too
                }
                PIO_MPI(MPI_Barrier(comm_)); // nobody opens it before removal ends
            }
        }

        log_summary(total_stats, field_stats, part_stats);
        return 0;
    }

private:
    std::filesystem::path chkpt_path(int chkpt_id, int step) const {
        std::ostringstream ss;
        ss << "chkpt_" << std::setw(6) << std::setfill('0') << chkpt_id << "_step_"
           << std::setw(6) << step << Backend::extension;
        return std::filesystem::path(cfg_.dir) / ss.str();
    }

    bool verify_checkpoint(const std::filesystem::path& path) {
        Field3d<double> dst(domain_);
        backend_.read_field(path, domain_, dst);

        const int g = cfg_.ghost, r = cfg_.sample_rate;
        long long local_bad = 0, local_seen = 0;
        for (int i = g; i < g + domain_.local[0]; i += r)
            for (int j = g; j < g + domain_.local[1]; j += r)
                for (int k = g; k < g + domain_.local[2]; k += r) {
                    ++local_seen;
                    if (field_.at(i, j, k) != dst.at(i, j, k))
                        ++local_bad;
                }

        long long bad = 0, seen = 0;
        PIO_MPI(MPI_Allreduce(&local_bad, &bad, 1, MPI_LONG_LONG, MPI_SUM, comm_));
        PIO_MPI(MPI_Allreduce(&local_seen, &seen, 1, MPI_LONG_LONG, MPI_SUM, comm_));
        if (rank_ == 0 && !cfg_.quiet)
            std::cout << "  verify: " << seen << " cells sampled, " << bad
                      << " mismatches\n";
        return bad == 0;
    }

    void log_banner() const {
        if (rank_ != 0)
            return;
        std::cout << "\n"
                  << std::string(72, '=') << "\n"
                  << "pio_bench -- PIC-like parallel I/O benchmark ("
                  << Backend::name << ")\n"
                  << std::string(72, '=') << "\n"
                  << "ranks: " << size_of(comm_) << " grid: " << domain_.dims[0] << 'x'
                  << domain_.dims[1] << 'x' << domain_.dims[2] << '\n'
                  << "global field: " << domain_.global[0] << 'x' << domain_.global[1]
                  << 'x' << domain_.global[2] << " ("
                  << std::fixed << std::setprecision(2)
                  << double(field_bytes_) * size_of(comm_) / (1024.0 * 1024.0 * 1024.0)
                  << " GiB/checkpoint)\n"
                  << "particles: " << particles_.global_count() << " total ("
                  << particles_.local_count() << " per rank, "
                  << std::setprecision(1)
                  << double(particle_bytes_) * size_of(comm_) / (1024.0 * 1024.0)
                  << " MiB/checkpoint)\n"
                  << cfg_.describe() << '\n'
                  << std::string(72, '=') << "\n\n";
    }

    void append_metric(int chkpt_id, int step, const std::filesystem::path& path,
                       double total_s, const WriteSplit& split, std::uint64_t bytes) {
        std::ofstream out(std::filesystem::path(cfg_.dir) / "pio_metrics.jsonl",
                          std::ios::app);
        out << "{\"benchmark\":\"pio_bench\",\"benchmark_version\":\""
            << kBenchmarkVersion << "\",\"backend\":\"" << Backend::name
            << "\",\"chkpt\":" << chkpt_id
            << ",\"step\":" << step << ",\"path\":\"" << path.filename().string()
            << "\",\"bytes\":" << bytes << ",\"ranks\":" << size_of(comm_) << ",\"dims\":["
            << domain_.dims[0] << "," << domain_.dims[1] << "," << domain_.dims[2]
            << "],\"local\":[" << domain_.local[0] << "," << domain_.local[1] << ","
            << domain_.local[2] << "],\"ghost\":" << cfg_.ghost << ",\"particles_per_rank\":"
            << cfg_.particles_per_rank << ",\"seconds\":" << std::setprecision(6) << total_s
            << ",\"field_s\":" << split.field_seconds << ",\"particle_s\":"
            << split.particle_seconds << ",\"mib_s\":"
            << double(bytes) / total_s / (1024.0 * 1024.0);
        if (cfg_.aggregators)
            out << ",\"aggregators\":" << *cfg_.aggregators;
        if (cfg_.buffer_bytes)
            out << ",\"buffer_bytes\":" << *cfg_.buffer_bytes;
        out << ",\"seed\":" << cfg_.seed << "}\n";
    }

    void log_summary(const RunningStats& total, const RunningStats& field,
                     const RunningStats& part) const {
        if (rank_ != 0 || total.n == 0)
            return;
        const double mib = double(field_bytes_ + particle_bytes_) / (1024.0 * 1024.0);
        std::cout << "\n"
                  << std::string(72, '=') << "\n"
                  << "pio_bench results (" << Backend::name << ", " << size_of(comm_)
                  << " ranks, " << mib << " MiB/checkpoint)\n"
                  << std::string(72, '=') << "\n"
                  << "wall s/checkpoint : " << total.fmt() << '\n'
                  << "write MiB/s       : mean=" << mib / total.mean
                  << " (best " << mib / total.min << ", worst " << mib / total.max << ")\n"
                  << "field phase s     : " << field.fmt() << '\n'
                  << "particle phase s  : " << part.fmt() << '\n'
                  << std::string(72, '=') << "\n\n";
    }

    BenchConfig cfg_;
    MPI_Comm comm_;
    int rank_;
    Domain domain_;
    Field3d<double> field_;
    ParticleSet particles_;
    Backend backend_;
    std::uint64_t field_bytes_{0};
    std::uint64_t particle_bytes_{0};
};

} // namespace pio
