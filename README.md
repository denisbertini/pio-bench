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
- [openPMD-api](https://github.com/openPMD/openPMD-api) (for `pmd_hdf5` / `pmd_adios2`)
- [ADIOS2](https://github.com/ornladios/ADIOS2) (optional, for `adios2` / `pmd_adios2`)

Missing optional dependencies simply omit the corresponding backends at build
time.

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

## Documentation

- Backend details: [`docs/backends.md`](docs/backends.md)
- Internal optimizations (MPI-IO 64-bit limits, HDF5 chunk layouts, ADIOS2
  telemetry overhead): [`docs/internals.md`](docs/internals.md)
- Contribution guidelines: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
