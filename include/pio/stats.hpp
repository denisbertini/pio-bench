// pio/stats.hpp -- timing + descriptive statistics.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>

namespace pio {

class Stopwatch {
public:
    Stopwatch() { restart(); }
    void restart() { t0_ = Clock::now(); }
    /// seconds since construction/restart
    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - t0_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0_;
};

struct RunningStats {
    std::size_t n{0};
    double mean{0}, m2{0}; // Welford
    double min{0}, max{0}, sum{0};

    void add(double x) {
        if (n == 0)
            min = max = x;
        else {
            min = std::min(min, x);
            max = std::max(max, x);
        }
        sum += x;
        ++n;
        const double d = x - mean;
        mean += d / static_cast<double>(n);
        m2 += d * (x - mean);
    }
    double stdev() const {
        return n > 1 ? std::sqrt(m2 / static_cast<double>(n - 1)) : 0.0;
    }
    double sem() const {
        return n > 0 ? stdev() / std::sqrt(static_cast<double>(n)) : 0.0;
    }
    std::string fmt() const {
        std::ostringstream os;
        os << "mean=" << mean << " sd=" << stdev() << " sem=" << sem()
           << " min=" << min << " max=" << max << " n=" << n;
        return os.str();
    }
};

} // namespace pio
