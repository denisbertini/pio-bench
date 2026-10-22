# pio-bench

An MPI parallel I/O benchmark that mimics the checkpoint and data patterns of
Particle-in-Cell (PIC) codes such as
[WarpX](https://github.com/BLAST-WarpX/warpx) and
[PIConGPU](https://github.com/ComputationalRadiationPhysics/picongpu). It
writes 3-D field matrices and particle arrays simultaneously into a single
shared file using multiple interchangeable I/O backends.

## Backends

`mpiio` · `hdf5` · `adios2` · `pmd_hdf5` · `pmd_adios2`

Backends that are not available at compile time are safely omitted. See
[`docs/backends.md`](docs/backends.md) for transport engines, file formats, and
technical notes.

## Requirements

- MPI implementation (OpenMPI / MPICH / Intel MPI)
- C++20 compiler (`mpicxx`)
- CMake ≥ 3.22
- Parallel [HDF5](https://www.hdfgroup.org/solutions/hdf5/) (for `hdf5` / `pmd_hdf5`)
- [ADIOS2](https://github.com/ornladios/ADIOS2) (for `adios2` / `pmd_adios2`)
- [openPMD-api](https://github.com/openPMD/openPMD-api) — optional, `pmd_hdf5` / `pmd_adios2` only

A missing library simply omits the backends that need it at build time. The
general backends (`mpiio`, `hdf5`, `adios2`) and the CI gate
(`scripts/ci_test.sh`) therefore need only parallel HDF5 and ADIOS2 — not
openPMD-api.

## Build

```bash
cmake -B build -DCMAKE_CXX_COMPILER=mpicxx \
      -DPIO_WITH_HDF5=ON \
      -DPIO_WITH_ADIOS2=ON
cmake --build build -j
```

## Quick Start

```bash
mpirun -n 64 ./build/pio_bench \
  --backend mpiio \
  --local 64 \
  --particles 100000 \
  --verify
```

`--help` lists all options. Metrics are logged to stdout and to
`pio_metrics.jsonl` in the output directory (`--dir`).

For Lustre and similar HPC filesystems, configure striping and tuning hints
(e.g. ROMIO hints, openPMD options) before running — see
[`docs/internals.md`](docs/internals.md). Automated test scripts and Slurm
batch files are provided under `scripts/`.

## CI / environment check

Because each backend drives a different parallel I/O library — MPI-IO/ROMIO,
parallel HDF5, and ADIOS2 — a run that completes and passes `--verify` is a
real integration test that the I/O stack is present, correctly built (parallel,
not serial), and wired to the MPI stack and the filesystem. A backend that
fails, or is silently compiled out, pinpoints a broken or missing dependency.

```sh
cd scripts && sbatch ci_test.sh    # 1 node, 4 ranks: mpiio, hdf5 (parallel), adios2, --verify
```

`ci_test.sh` runs the three major HPC I/O backends — `mpiio`, `hdf5`
(parallel) and `adios2` — at a minimal size, to test the proper installation
of the backends and their MPI dependencies. It can be used to quickly validate
the I/O stack installed on the compute nodes of a cluster, or in a container
image environment, before trusting it for production. For a full multi-node
check — 4 nodes, all five backends, real Lustre striping — use
`scripts/submit_all.sh`.

## Documentation

- Backend details: [`docs/backends.md`](docs/backends.md)
- Internal optimizations (MPI-IO 64-bit limits, HDF5 chunk layouts, ADIOS2
  telemetry overhead): [`docs/internals.md`](docs/internals.md)
- Contribution guidelines: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
