// pio_bench.cpp -- entry point: parse parameters once, instantiate the
// benchmark for the selected backend exactly once (compile-time dispatch
// over concept-checked policy types; unavailable backends never get
// instantiated thanks to `if constexpr`).
#include <exception>
#include <iostream>

#include <mpi.h>

#include "pio/benchmark.hpp"

using namespace pio;

namespace {

template <typename Backend>
int run_selected(const BenchConfig& cfg) {
    if constexpr (Backend::available) {
        return PicBenchmark<Backend>(cfg, MPI_COMM_WORLD).run();
    } else {
        // Exit code 2 = "not available in this build" (smoke.sh maps it to
        // SKIPPED; also matched by message text as a belt-and-braces).
        if (rank_of(MPI_COMM_WORLD) == 0)
            std::cerr << "-I- : SKIPPED: " << Backend::name
                      << " is not compiled into this build"
                         " (library unavailable or built without MPI)\n";
        return 2;
    }
}

} // namespace

int main(int argc, char** argv) {
    BenchConfig cfg;
    try {
        cfg = BenchConfig::parse(argc, argv);
    } catch (const HelpRequested&) {
        std::cout << BenchConfig::usage_text;
        return 0;
    } catch (const std::invalid_argument& e) {
        std::cerr << "-E- : " << e.what() << "\n\n" << BenchConfig::usage_text;
        return 2;
    }

    PIO_MPI(MPI_Init(&argc, &argv));
    int rc = 0;
    try {
        switch (cfg.backend) {
        case BackendKind::Mpiio:
            rc = run_selected<MpiIoBackend>(cfg);
            break;
        case BackendKind::Hdf5:
            rc = run_selected<Hdf5Backend>(cfg);
            break;
        case BackendKind::Adios2:
            rc = run_selected<Adios2Backend>(cfg);
            break;
        }
    } catch (const std::exception& e) {
        std::cerr << "-E- : [" << rank_of(MPI_COMM_WORLD) << "] " << e.what() << "\n";
        rc = 1;
    }
    MPI_Finalize();
    return rc;
}
