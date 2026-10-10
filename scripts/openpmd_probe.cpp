// scripts/openpmd_probe.cpp -- standalone openPMD API probe, OUTSIDE CMake's
// macro-escaping layers. Compile against the openPMD you actually have:
//
//   mpicxx -std=c++20 scripts/openpmd_probe.cpp -lopenPMD -o /tmp/pmd_probe \
//       && echo "openPMD current API OK"
//   mpicxx -std=c++20 -DPIO_PMD_LEGACY_CTOR scripts/openpmd_probe.cpp \
//       -lopenPMD -o /tmp/pmd_probe && echo "openPMD legacy API OK"
//
// (add -I/-L flags if openPMD lives outside the default search paths)
//
// Exercises the EXACT call surface pio_bench's pmd_* backends use against
// openPMD 0.17.x: Series(path, access, comm, options-JSON), Access::CREATE,
// determineDatatype, mesh component storeChunk(vector&), scalar particle
// records via the PatchRecordComponent(BaseRecord) view ctor, component
// storeChunk + loadChunkRaw, flush. If this compiles and CMake still says
// compiled-OUT, the CMake probe is broken -- not your openPMD.
#include <openPMD/openPMD.hpp>

#include <mpi.h>

#include <string>
#include <vector>

int main() {
#ifdef PIO_PMD_LEGACY_CTOR
    openPMD::Series s("probe_unused.h5", openPMD::Access::CREATE,
                      std::string("HDF5"), MPI_COMM_WORLD);
#else
    openPMD::Series s("probe_unused.h5", openPMD::Access::CREATE,
                      MPI_COMM_WORLD, std::string(R"({"backend":"hdf5"})"));
#endif
    auto it = s.iterations[0];
    std::vector<double> buf(2, 1.0);

    // mesh: record/component + dataset + contiguous-container storeChunk
    auto comp = it.meshes["m"]["c"];
    comp.resetDataset(openPMD::Dataset(openPMD::determineDatatype<double>(),
                                       openPMD::Extent{2}));
    comp.storeChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});

    // particles: scalar record via component view, and component leaf
    auto scalar = it.particles["sp"]["mass"];
    openPMD::PatchRecordComponent sc(scalar);
    sc.resetDataset(openPMD::Dataset(openPMD::determineDatatype<double>(),
                                     openPMD::Extent{2}));
    sc.storeChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});

    auto px = it.particles["sp"]["position"]["x"];
    px.resetDataset(openPMD::Dataset(openPMD::determineDatatype<double>(),
                                     openPMD::Extent{2}));
    px.storeChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});
    px.loadChunkRaw(buf.data(), openPMD::Offset{0}, openPMD::Extent{2});

    s.flush();
    return 0;
}
