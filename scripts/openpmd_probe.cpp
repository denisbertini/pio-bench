// scripts/openpmd_probe.cpp -- standalone openPMD API probe, OUTSIDE CMake's
// macro-escaping layers (which corrupted the in-configure probe the first
// two times). Compile against the openPMD you actually have:
//
//   mpicxx -std=c++20 scripts/openpmd_probe.cpp -lopenPMD -o /tmp/pmd_probe \
//       && echo "openPMD current API OK"
//   mpicxx -std=c++20 -DPIO_PMD_LEGACY_CTOR scripts/openpmd_probe.cpp \
//       -lopenPMD -o /tmp/pmd_probe && echo "openPMD legacy API OK"
//
// (add -I/-L flags if openPMD lives outside the default search paths)
//
// Exercises the EXACT call surface pio_bench's pmd_* backends use:
// Series ctor (both forms), iterations, mesh record component, scalar vs
// component particle records, void* storeChunk/loadChunk, typed loadChunk,
// flush. If this compiles and CMake still says compiled-OUT, the CMake
// probe is broken -- not your openPMD.
#include <openPMD/openPMD.hpp>

#include <mpi.h>

#include <string>

int main() {
#ifdef PIO_PMD_LEGACY_CTOR
    openPMD::Series s("probe_unused.h5", openPMD::Access::Create,
                      std::string("HDF5"), MPI_COMM_WORLD);
#else
    openPMD::Series s("probe_unused.h5", openPMD::Access::Create,
                      MPI_COMM_WORLD, std::string(R"({"backend":"hdf5"})"));
#endif
    auto it = s.iterations[0];

    double buf[2] = {1.0, 2.0};

    // mesh: record/component + dataset + chunk + raw-pointer store/load
    auto comp = it.meshes["m"]["c"];
    comp.resetDataset(openPMD::Dataset(openPMD::determineType<double>(),
                                       openPMD::Shape{2}));
    comp.setChunkSize(openPMD::Shape{1});
    comp.storeChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});
    (void)comp.loadChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});

    // particles: scalar record (writable directly) and component leaf
    auto scalar = it.particles["sp"]["mass"];
    scalar.resetDataset(openPMD::Dataset(openPMD::determineType<double>(),
                                         openPMD::Shape{2}));
    scalar.storeChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});
    auto px = it.particles["sp"]["position"]["x"];
    px.resetDataset(openPMD::Dataset(openPMD::determineType<double>(),
                                     openPMD::Shape{2}));
    px.storeChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});
    (void)px.loadChunk<double>(openPMD::Offset{0}, openPMD::Extent{2});

    s.flush();
    return 0;
}
