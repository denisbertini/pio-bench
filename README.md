# pio-bench

A modern C++20 MPI I/O benchmark mimicking a Particle-In-Cell (PIC) code's
checkpoint pattern: 3-D field matrices **plus** particles written to a single
shared parallel file, with selectable backends:

| backend | container | file |
|---|---|---|
| `mpiio`  | raw MPI-IO (ROMIO hints) | `.bin` |
| `hdf5`   | Parallel HDF5           | `.h5`  |
| `adios2` | ADIOS2 BP5              | `.bp`  |

Rewritten from the ground up out of the legacy `ci_ompi` test framework
(`ci_pic` + `ci_pic_chkpt`, merged into one class hierarchy).

## What it measures

Every checkpoint writes, into **one** file:

* **fields** — one `double` matrix per rank in a global 3-D subarray layout
  (ghost cells excluded from the file, exactly like a real PIC dump);
* **particles** — `{x, y, z, px, py, pz, mass, id}` per particle, generated
  deterministically per rank (seeded by `seed, rank`), merged into the same
  file as the fields via MPI derived record types (MPI-IO), strided
  hyperslabs out of the AoS (HDF5), or `LocalValueDim` variables (ADIOS2).

Per-phase timings (fields vs particles) and throughput are reported live and
appended as machine-readable **JSON Lines** (`pio_metrics.jsonl`), every line
stamped with `benchmark_version` so results always identify the binary that
produced them.

## Build

```sh
cmake -B build -DCMAKE_CXX_COMPILER=mpicxx \
      [-DPIO_WITH_HDF5=ON] [-DPIO_WITH_ADIOS2=ON]
cmake --build build -j
```

Requirements: C++20 compiler, MPI. HDF5 must be a **parallel** build (serial
HDF5 is auto-detected and the backend compiled out). ADIOS2 optional.
Backends not available at compile time are removed by `if constexpr` — they
never exist in the binary.

## Run

```sh
mpirun -n 64 ./build/pio_bench \
    --local 64 --ghost 1 \
    --particles 100000 --seed 42 \
    --backend mpiio --aggregators 8 --buffer 4MiB \
    --steps 20 --interval 2 --compute-ms 500 \
    --verify --sample-rate 16
```

`--help` lists all options. `--verify` reads the checkpoint back and
sample-compares cells; a mismatch **aborts the job** (a benchmark that
silently corrupts is worse than none).

Under Darshan (fitness instrumentation):

```sh
DARSHAN_LOGPATH=$PWD darshan-runtime mpirun -n 64 ./build/pio_bench ...
```

## Design (vs. the legacy ci_ompi code)

* **One class, many backends**: `PicBenchmark<Backend>` parameterized over
  concept-checked (`pio::IoBackend`) policy types. The old code replicated
  an `enum`+`if` ladder through six methods and a bit-flag config enum.
* **RAII MPI**: `Info`/`Datatype`/`File` wrappers, typed exceptions with
  source location. The old `CHECK_ERR` macro printed and *continued* after
  failed I/O — silent corruption by design.
* **Contiguous `Field3d`** replaces `double***`; the ghost-aware memory
  subarray/hyperslab views are derived from it, fixing a real bug in the old
  HDF5 writer (a simple local-dims memspace with an `&arr[g][g][g]` pointer
  wrote **wrong data for any `ghost > 0`**: padded interior rows are not
  contiguous).
* **Typed parameters**: validated, self-describing `BenchConfig`;
  `std::optional` for "unset" hints instead of `0 means default`.
* **Deterministic payload**: field values are the flattened global index;
  particle `id`s are the globally unique rank-merged index.

## Intended integration (mpiio_evolve)

Candidate JSON maps 1:1 onto flags — `romio.cb_nodes → --aggregators`,
`romio.cb_buffer_size → --buffer`, `engine → --backend` — and the JSONL
output feeds the measurement ledger directly. Built inside the compute
container image, pinned by release tag, so the fitness instrument is frozen
per campaign.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
