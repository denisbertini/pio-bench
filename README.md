# pio-bench

MPI parallel I/O benchmark mimicking a Particle-In-Cell (PIC) checkpoint:
3-D field matrices and particles written to a single shared file, with
selectable backends:

| backend | transport | file |
|---|---|---|
| `mpiio`  | MPI-IO (ROMIO hints) | `.bin` |
| `hdf5`   | Parallel HDF5        | `.h5`  |
| `adios2` | ADIOS2 BP5           | `.bp`  |

## What it measures

Every checkpoint writes, into **one** file:

* **fields** — one `double` matrix per rank in a global 3-D subarray
  layout, ghost cells excluded from the file;
* **particles** — `{x, y, z, px, py, pz, mass, id}` per particle, generated
  deterministically per rank (seeded by `seed, rank`) and merged into the
  same file across all ranks.

Per-phase timings (fields vs particles) and throughput are reported live
and appended as machine-readable JSON Lines (`pio_metrics.jsonl`), each
line stamped with `benchmark_version`.

## Build

Requirements: MPI, a C++20 compiler. HDF5 must be a **parallel** build
(serial HDF5 is auto-detected and the backend compiled out). ADIOS2 is
optional; unavailable backends are removed at compile time.

```sh
cmake -B build -DCMAKE_CXX_COMPILER=mpicxx \
      [-DPIO_WITH_HDF5=ON] [-DPIO_WITH_ADIOS2=ON]
cmake --build build -j
```

## Run

```sh
mpirun -n 64 ./build/pio_bench \
    --local 64 --ghost 1 \
    --particles 100000 --seed 42 \
    --backend mpiio --aggregators 8 --buffer 4MiB \
    --steps 20 --interval 2 --compute-ms 500 \
    --verify --sample-rate 16
```

`--help` lists all options. `--verify` reads the checkpoint back — field
cells sampled, particle blocks fully — and a mismatch aborts the job.

Under Darshan:

```sh
DARSHAN_LOGPATH=$PWD darshan-runtime mpirun -n 64 ./build/pio_bench ...
```

## Test

```sh
scripts/smoke.sh        # runs every compiled-in backend with --verify
```

Backends absent from the build report `SKIPPED`, not failure. Tunables:
`RANKS`, `LOCAL`, `PARTICLES`, `STEPS`, `MPIRUN` (see script header).

## Notes

* Field values are the flattened global index; particle `id`s are the
  globally unique merged index — both cheap to verify independently.
* MPI-IO writes particles directly from the in-memory array of structs
  (MPI derived record type, zero copies). HDF5 stores per-member datasets:
  by default each member is pre-packed into a contiguous staging vector
  so `H5Dwrite` takes its fast path; `--h5-strided` instead hands HDF5 the
  AoS stride pattern, producing byte-identical files — the gap between
  the two quantifies the in-library conversion cost. ADIOS2 pre-packs
  per member into explicit global-shape variables.
* ADIOS2/BP5 computes per-block min/max/sum statistics on every `Put` by
  default (StatsLevel 2) — a full extra pass over the payload on the writer
  CPU, the leading suspect for BP5's write-rate gap vs. raw MPI-IO.
  `--no-adi-stats` sets StatsLevel 0 to isolate that cost; the file then
  carries no block statistics (affects later query/transform tools, not reads).
* With `ghost > 0` the ADIOS2 backend gathers the field interior into a
  temporary block (BP5 `Put` requires contiguous data).

## Integration (mpiio_evolve)

The tuning harness injects candidates via `ROMIO_HINTS` (and
`lfs setstripe` on the data directory), so pio-bench must not pre-set
MPI_Info keys: with no `--aggregators/--buffer/--no-cb` flags its Info is
empty and the environment's hints take full effect. The JSONL output is
the measurement feed; built inside the compute container image, pinned
by release tag.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
