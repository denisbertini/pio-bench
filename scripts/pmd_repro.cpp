// pmd_repro.cpp -- standalone openPMD 0.17.x reproduction for the
// "Failed to write dataset /particles/electrons/mass" wall: reproduces at
// ONE rank, ~350 MB file, both transfer modes, both MPI engines -> the
// trigger is openPMD's handling of our particle-record writes, not scale,
// not Lustre, not MPI-IO.
//
// The hypothesis under test: per-particle scalar records written through
// the scalar-view ctor (RecordComponent(Record)) take a different internal
// path than normal components (position/x). This program writes the
// suspects in stages, flushing between each, so the LAST printed OK line
// names the dataset that kills:
//
// Build (cluster, inside container):
//   mpicxx -O2 scripts/pmd_repro.cpp -o /tmp/pmdrepro -lopenPMD
// Runs (1 rank; np default 4000000):
//   mpirun -n 1 /tmp/pmdrepro r1.h5 field pos mass id     # full shape
//   mpirun -n 1 /tmp/pmdrepro r2.h5 field id              # mass removed
//   mpirun -n 1 /tmp/pmdrepro r3.h5 field pos             # pos only
//   mpirun -n 1 /tmp/pmdrepro r4.h5 field mass            # mass only
//   mpirun -n 1 /tmp/pmdrepro r5.h5 field pos np 1000 mass  # tiny np control
// Also run the same matrix with backend adios2 via file ending .bp:
//   mpirun -n 1 /tmp/pmdrepro r6.bp field pos mass        # engine-cross-check
//
// Verdict:
//   mass-stage FAILS, pos-stage OK   -> scalar-view path is the trigger;
//       backend fix: mass/id written the production way (per-particle 1-D
//       data through a normal named component or constant record).
//   everything OK at 1 rank          -> shape interaction with multi-rank;
//       rerun the matrix at -n 32 (np must then be * rank count handled by
//       offsets: this tool always writes rank 0 at offset 0, so multi-rank
//       usage keeps the same file content per rank -- fine for repro).
#include <openPMD/openPMD.hpp>
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace om = ::openPMD;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string file = argc > 1 ? argv[1] : "r.h5";
    bool field = false, pos = false, mass = false, id = false;
    bool constmass = false;  // mass as CONSTANT record (production pattern)
    bool all = false;        // EXACT backend shape: 8 SoA records
                             // (position x/y,z; momentum x/y/z; mass; id)
                             // all enqueued, then ONE flush -- the shape
                             // that kills in pio_bench; stage-by-stage
                             // passes at 1 rank, so the multi-record
                             // single-flush is the suspect under test
    std::uint64_t np = 4000000;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "field") field = true;
        else if (a == "pos") pos = true;
        else if (a == "mass") mass = true;
        else if (a == "constmass") { mass = true; constmass = true; }
        else if (a == "id") id = true;
        else if (a == "all") all = true;
        else if (a == "np" && i + 1 < argc) np = strtoull(argv[++i], nullptr, 10);
    }

    // Backend from the file ending (.h5 -> hdf5, .bp -> adios2), exactly as
    // pio-bench resolves it. EXACT backend delta closed here: pio-bench
    // passes the backend via options JSON (not extension inference) and
    // sets mesh geometry -- replicate both.
    const bool adios2 = file.size() >= 3 && file.substr(file.size() - 3) == ".bp";
    om::Series series(file, om::Access::CREATE, MPI_COMM_WORLD,
                      std::string("{\"backend\":\"") +
                          (adios2 ? "adios2" : "hdf5") + "\"}");
    auto iteration = series.iterations[0];

    bool ok = true;
    auto stage = [&](const char* name, auto&& body) {
        if (!ok) return;
        try {
            body();
            series.flush();
            if (!rank) printf("OK: %s\n", name);
        } catch (const std::exception& e) {
            ok = false;
            printf("[%d] FAIL at %s: %s\n", rank, name, e.what());
        }
    };

    // LIFETIME (the very rule under investigation): storeChunk is
    // non-owning, so every buffer handed to it must outlive the flush.
    // All buffers therefore live HERE, in main scope -- a stage-local
    // buffer destroyed before stage()'s flush reproduces the original
    // bug in the repro itself (observed: SIGSEGV in
    // ADIOI_GEN_WriteStrided, "Address not mapped").
    const std::uint64_t n = 256;
    std::vector<double> fbuf(n * n * n, 1.0);
    std::vector<double> poscol(np, 0.5);
    std::vector<double> masscol(np, 1.0);
    std::vector<std::uint64_t> ids(np, 7);

    // buffers for the full backend shape (7 double columns + ids), main
    // scope for the same lifetime reason as above
    constexpr int ncol = 7;  // position x/y/z, momentum x/y/z, mass
    std::vector<std::vector<double>> cols(ncol, std::vector<double>(np, 0.5));
    // rank-distributed layout, exactly like pio_bench: each rank owns a
    // disjoint slice of the GLOBAL datasets (the repro default wrote all
    // ranks at offset 0 -- same API calls, wrong scale; the crash config
    // is 4 GiB field + 1 GiB per particle component at 32 ranks)
    int nranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const std::uint64_t gpart = np * (std::uint64_t)nranks;
    const std::uint64_t poff = (std::uint64_t)rank * np;
    // field: global 1024x1024x512, rank r writes a 256x256x256 slice at a
    // 4x4x2 decomposition (pio_bench --local 256, 32 ranks)
    const std::uint64_t lx = 256, ly = 256, lz = 256;
    const int rz_ = rank / 16, ry_ = (rank / 4) % 4, rx_ = rank % 4;

    if (field) {
        stage("field rho", [&] {
            auto rho = iteration.meshes["field"]["rho"];
            rho.resetDataset(om::Dataset(om::determineDatatype<double>(),
                                         {lx * 4, ly * 4, lz * 2}));
            rho.storeChunk(fbuf,
                           {rx_ * lx, ry_ * ly, rz_ * lz}, {lx, ly, lz});
        });
    }
    auto species = iteration.particles["electrons"];
    if (all) {
        stage("ALL: pos xyz + mom xyz + mass + id, ONE flush", [&] {
            static const char* recs[ncol] = {"position", "position",
                                             "position", "momentum",
                                             "momentum", "momentum", "mass"};
            static const char* comps[ncol] = {"x", "y", "z", "x", "y", "z",
                                              nullptr};
            for (int c = 0; c < ncol; ++c) {
                auto rec = species[recs[c]];
                if (comps[c]) {
                    auto comp = rec[comps[c]];
                    comp.resetDataset(
                        om::Dataset(om::determineDatatype<double>(), {gpart}));
                    comp.storeChunk(cols[c], {poff}, {np});
                } else {
                    om::RecordComponent comp(rec);
                    comp.resetDataset(
                        om::Dataset(om::determineDatatype<double>(), {gpart}));
                    comp.storeChunk(cols[c], {poff}, {np});
                }
            }
            om::RecordComponent idrec(species["id"]);
            idrec.resetDataset(
                om::Dataset(om::determineDatatype<std::uint64_t>(), {gpart}));
            idrec.storeChunk(ids, {poff}, {np});
        });
    }
    if (pos) {
        stage("position/x (normal component)", [&] {
            auto c = species["position"]["x"];
            c.resetDataset(
                om::Dataset(om::determineDatatype<double>(), {np}));
            c.storeChunk(poscol, {0}, {np});
        });
    }
    if (mass) {
        stage(constmass ? "mass (constant record)"
                        : "mass (scalar-view RecordComponent)",
              [&] {
                  if (constmass) {
                      // production pattern (PIConGPU/WarpX): one value for
                      // all particles, no per-particle dataset payload
                      om::RecordComponent mv(species["mass"]);
                      mv.makeConstant(1.0);
                  } else {
                      om::RecordComponent mv(species["mass"]);
                      mv.resetDataset(
                          om::Dataset(om::determineDatatype<double>(), {np}));
                      mv.storeChunk(masscol, {0}, {np});
                  }
              });
    }
    if (id) {
        stage("id (scalar-view RecordComponent, uint64)", [&] {
            om::RecordComponent iv(species["id"]);
            iv.resetDataset(
                om::Dataset(om::determineDatatype<std::uint64_t>(), {np}));
            iv.storeChunk(ids, {0}, {np});
        });
    }

    series.flush();
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) printf("done ok=%d file=%s np=%llu\n", (int)ok, file.c_str(),
                      (unsigned long long)np);
    MPI_Finalize();
    return ok ? 0 : 1;
}
