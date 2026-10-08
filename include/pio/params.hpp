// pio/params.hpp -- modern parameter handling.
//
// Replaces the old ci_ompi_test_config: one getopt soup struct with
// bit-flag enum (CI_BIT(i) ...), loose ints and "0 means default".
// Here: strongly typed value semantics, std::optional for "unset"
// (aggregators/buffer hints), validated on parse, self-describing.
//
// Parse contract: BenchConfig::parse() throws std::invalid_argument with a
// human-readable reason; main() turns that into usage + exit code 2.
// (--help throws HelpRequested to break that flow out cleanly.)
#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pio {

enum class BackendKind { Mpiio, Hdf5, Adios2 };

inline std::string_view to_string(BackendKind b) {
    switch (b) {
    case BackendKind::Mpiio:
        return "mpiio";
    case BackendKind::Hdf5:
        return "hdf5";
    case BackendKind::Adios2:
        return "adios2";
    }
    return "?";
}

inline std::optional<BackendKind> backend_from_string(std::string_view s) {
    if (s == "mpiio" || s == "mpi-io")
        return BackendKind::Mpiio;
    if (s == "hdf5" || s == "phdf5")
        return BackendKind::Hdf5;
    if (s == "adios2" || s == "bp5")
        return BackendKind::Adios2;
    return std::nullopt;
}

struct HelpRequested {};

class ParseError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

namespace detail {

// "512", "8M", "4MiB", "1G" -> bytes (1024-based). Plain integers allowed.
inline std::uint64_t parse_size(std::string_view s) {
    if (s.empty())
        throw ParseError("empty size value");
    std::uint64_t mult = 1;
    if (s.size() > 1) {
        using namespace std::string_view_literals;
        for (auto [suffix, m] : {std::pair{"KiB"sv, 1024ull}, {"MiB"sv, 1024ull * 1024},
                                 {"GiB"sv, 1024ull * 1024 * 1024},
                                 {"K"sv, 1024ull}, {"M"sv, 1024ull * 1024},
                                 {"G"sv, 1024ull * 1024 * 1024}}) {
            if (s.ends_with(suffix)) {
                mult = m;
                s = s.substr(0, s.size() - suffix.size());
                break;
            }
        }
    }
    std::uint64_t v = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size())
        throw ParseError("invalid size: '" + std::string(s) + "'");
    return v * mult;
}

inline long long parse_ll(std::string_view s, std::string_view key) {
    long long v = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size())
        throw ParseError("invalid integer for " + std::string(key) + ": '" +
                         std::string(s) + "'");
    return v;
}

inline double parse_d(std::string_view s, std::string_view key) {
    double v = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size())
        throw ParseError("invalid number for " + std::string(key) + ": '" +
                         std::string(s) + "'");
    return v;
}

} // namespace detail

struct BenchConfig {
    // ---- grid / decomposition ------------------------------------------------
    std::array<int, 3> local{64, 64, 64}; // interior cells per rank (x,y,z)
    int ghost{1};

    // ---- particles -------------------------------------------------------------
    std::size_t particles_per_rank{100'000};
    std::uint64_t seed{42};

    // ---- I/O ------------------------------------------------------------------
    BackendKind backend{BackendKind::Mpiio};
    std::string dir{"."};
    std::optional<int> aggregators{};         // cb_nodes / NumAggregators
    std::optional<std::size_t> buffer_bytes{}; // cb_buffer_size / BufferVSize
    bool collective_buffering{true};           // ROMIO hint (mpiio/hdf5)

    // ---- time loop ------------------------------------------------------------
    int steps{10};
    int chkpt_interval{1};
    int max_checkpoints{-1}; // <0: unlimited
    double compute_ms{0.0};  // simulated compute per step (sleep)

    // ---- validation / hygiene ---------------------------------------------------
    bool verify{false};
    int sample_rate{16};  // verify every Nth cell per axis
    bool keep_files{false};
    bool quiet{false};

    static constexpr const char* usage_text =
        "pio_bench -- PIC-like MPI-IO/HDF5/ADIOS2 checkpoint benchmark\n"
        "\n"
        "  --local N | Nx,Ny,Nz   interior cells per rank        (64,64,64)\n"
        "  --ghost N              ghost cells per side            (1)\n"
        "  --particles N          particles per rank (suffix ok)  (100k->100000)\n"
        "  --seed N               RNG seed                        (42)\n"
        "  --backend NAME         mpiio | hdf5 | adios2           (mpiio)\n"
        "  --dir PATH             output directory                (.)\n"
        "  --aggregators N        cb_nodes / NumAggregators       (ROMIO default)\n"
        "  --buffer SIZE          cb_buffer_size / BufferVSize,   (library default)\n"
        "                         suffixes: K M G or KiB MiB GiB\n"
        "  --no-cb                disable ROMIO collective buffering\n"
        "  --steps N              timesteps                       (10)\n"
        "  --interval N           checkpoint every N steps        (1)\n"
        "  --max-checkpoints N    cap checkpoints, -1 unlimited   (-1)\n"
        "  --compute-ms MS        sleep per step (fake compute)   (0)\n"
        "  --verify               read back and sample-compare    (off)\n"
        "  --sample-rate N        verify stride per axis          (16)\n"
        "  --keep-files           keep checkpoint files           (remove)\n"
        "  --quiet                suppress per-step output\n"
        "  --help                 this text\n";

    /// Accepts `--key value` and `--key=value`.
    static BenchConfig parse(int argc, char** argv) {
        BenchConfig cfg;
        auto need_value = [&](int& i, std::string_view key) -> std::string_view {
            std::string_view a = argv[i];
            if (auto eq = a.find('='); eq != std::string_view::npos)
                return a.substr(eq + 1);
            if (i + 1 >= argc)
                throw ParseError("missing value for " + std::string(key));
            return argv[++i];
        };

        for (int i = 1; i < argc; ++i) {
            std::string_view a = argv[i];
            std::string_view key = a.starts_with("--") ? a.substr(2) : a;
            auto strip = [&] {
                if (auto eq = key.find('='); eq != std::string_view::npos)
                    key = key.substr(0, eq);
            };
            strip();

            if (key == "help" || key == "h")
                throw HelpRequested{};
            else if (key == "local") {
                std::string_view v = need_value(i, "--local");
                std::array<int, 3> d{0, 0, 0};
                size_t idx = 0;
                size_t pos = 0;
                while (pos <= v.size() && idx < 3) {
                    auto comma = v.find(',', pos);
                    if (comma == std::string_view::npos)
                        comma = v.size();
                    d[idx++] = static_cast<int>(detail::parse_ll(v.substr(pos, comma - pos), "--local"));
                    pos = comma + 1;
                }
                if (idx == 1)
                    d[1] = d[2] = d[0];
                else if (idx != 3)
                    throw ParseError("--local expects N or Nx,Ny,Nz");
                cfg.local = d;
            } else if (key == "ghost") {
                cfg.ghost = static_cast<int>(detail::parse_ll(need_value(i, key), key));
            } else if (key == "particles") {
                cfg.particles_per_rank = static_cast<std::size_t>(detail::parse_size(need_value(i, key)));
            } else if (key == "seed") {
                cfg.seed = static_cast<std::uint64_t>(detail::parse_ll(need_value(i, key), key));
            } else if (key == "backend") {
                std::string_view v = need_value(i, key);
                auto b = backend_from_string(v);
                if (!b)
                    throw ParseError("unknown --backend '" + std::string(v) +
                                     "' (mpiio|hdf5|adios2)");
                cfg.backend = *b;
            } else if (key == "dir") {
                cfg.dir = std::string(need_value(i, key));
            } else if (key == "aggregators") {
                cfg.aggregators = static_cast<int>(detail::parse_ll(need_value(i, key), key));
            } else if (key == "buffer") {
                cfg.buffer_bytes = static_cast<std::size_t>(detail::parse_size(need_value(i, key)));
            } else if (key == "no-cb") {
                cfg.collective_buffering = false;
            } else if (key == "steps") {
                cfg.steps = static_cast<int>(detail::parse_ll(need_value(i, key), key));
            } else if (key == "interval") {
                cfg.chkpt_interval = static_cast<int>(detail::parse_ll(need_value(i, key), key));
            } else if (key == "max-checkpoints") {
                cfg.max_checkpoints = static_cast<int>(detail::parse_ll(need_value(i, key), key));
            } else if (key == "compute-ms") {
                cfg.compute_ms = detail::parse_d(need_value(i, key), key);
            } else if (key == "verify") {
                cfg.verify = true;
            } else if (key == "sample-rate") {
                cfg.sample_rate = static_cast<int>(detail::parse_ll(need_value(i, key), key));
            } else if (key == "keep-files") {
                cfg.keep_files = true;
            } else if (key == "quiet") {
                cfg.quiet = true;
            } else {
                throw ParseError("unknown option: " + std::string(a));
            }
        }

        // ---- invariants ---------------------------------------------------------
        for (int v : cfg.local)
            if (v <= 0)
                throw ParseError("--local dimensions must be > 0");
        if (cfg.ghost < 0)
            throw ParseError("--ghost must be >= 0");
        if (cfg.particles_per_rank == 0)
            throw ParseError("--particles must be > 0");
        if (cfg.steps <= 0)
            throw ParseError("--steps must be > 0");
        if (cfg.chkpt_interval <= 0)
            throw ParseError("--interval must be > 0");
        if (cfg.sample_rate <= 0)
            throw ParseError("--sample-rate must be > 0");
        if (cfg.aggregators && *cfg.aggregators < 0)
            throw ParseError("--aggregators must be >= 0");
        return cfg;
    }

    std::string describe() const {
        std::ostringstream os;
        os << "backend=" << to_string(backend)
           << " local=" << local[0] << 'x' << local[1] << 'x' << local[2]
           << " ghost=" << ghost << " steps=" << steps
           << " interval=" << chkpt_interval
           << " max_chkpts=" << max_checkpoints
           << " compute_ms=" << compute_ms << " particles/rank="
           << particles_per_rank << " seed=" << seed
           << " aggregators=" << (aggregators ? std::to_string(*aggregators) : "default")
           << " buffer=" << (buffer_bytes ? std::to_string(*buffer_bytes) + " B" : "default")
           << " cb=" << (collective_buffering ? "on" : "off")
           << " verify=" << (verify ? "on" : "off")
           << " keep_files=" << (keep_files ? "on" : "off");
        return os.str();
    }
};

} // namespace pio
