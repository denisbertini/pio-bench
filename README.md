# pio-bench

MPI parallel I/O benchmark mimicking a Particle-In-Cell (PIC) checkpoint:
3-D field matrices and particles written to a single shared file, with
selectable backends:

| backend | transport | file |
|---|---|---|
| `mpiio`  | MPI-IO (ROMIO hints) | `.bin` |
| `hdf5`   | Parallel HDF5        | `.h5`  |
| `adios2` | ADIOS2 BP5           | `.bp`  |
| `pmd_hdf5`   | openPMD-api → HDF5 | `.h5`  |
| `pmd_adios2` | openPMD-api → BP5  | `.bp`  |

The `pmd_*` backends write the **standard openPMD data model** (mesh
record `field/rho` + species `electrons` with SoA `position/momentum/mass/
id` + standard `positionOffset` constants) — production PIC checkpoint
semantics, still bit-exactly verified.  The write path mirrors openPMD's
own examples (8a + 3b at the 0.17.1 tag) call for call: owning
`shared_ptr` stores, scalar records used directly on the `Record`, one
flush at `iteration.close()`.  The AoS→SoA transpose is timed inside the
particle window; `--pmd-split` opts into two incremental flushes to
attribute field vs particle seconds honestly (openPMD stores are lazy —
with the canonical single flush the split reports 0 and only the wall
total is claimed).  Requires openPMD-api 0.16+/0.17.x (a probe falls back
to the pre-0.17 ctor; without openPMD the backends compile out).

## What it measures

Every checkpoint writes, into **one** file:

* **fields** — one `double` matrix per rank in a global 3-D subarray
  layout, ghost cells excluded from the file;
* **particles** — `{x, y, z, px, py, pz, mass, id}` per particle, generated
  deterministically per rank (seeded by `seed, rank`) and merged into the
  same file across all ranks.

Per-phase timings (fields vs particles; for `pmd_*` only with
`--pmd-split`, see above) and throughput are reported live
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

On Lustre, create the output directory striped — a single file lands on
ONE OST, capping every backend at one target's rate:

```sh
mkdir -p chkpts && lfs setstripe -c 8 chkpts/     # 8 OSTs; more for bigger jobs
HDF5_USE_FILE_LOCKING=FALSE mpirun -n 32 ./build/pio_bench \
    --backend pmd_hdf5 --dir ./chkpts ...          # safe: one job, write-once files
```

`pmd_hdf5` knobs: `pmd_chunks=rank` (default) aligns each dataset chunk
with the per-rank box (openPMD's auto-chunker caps at 4 MiB and shatters
the brick); `--pmd-auto-chunks` restores the library default for A/B.

Under Darshan:

```sh
DARSHAN_LOGPATH=$PWD darshan-runtime mpirun -n 64 ./build/pio_bench ...
```

## Test

```sh
scripts/smoke.sh          # runs every compiled-in backend with --verify
scripts/pmd_env_matrix.sh # pmd throughput attribution matrix (one mpirun
                          # per row: locking, transfer mode, allocation,
                          # metadata, chunking, ADIOS2 reference)
```

Backends absent from the build report `SKIPPED`, not failure. Tunables:
`RANKS`, `LOCAL`, `PARTICLES`, `STEPS`, `MPIRUN` (see script header).

Measured on Lustre/Virgo, 32 ranks on one node, 11.9 GiB/checkpoint,
`--verify` clean throughout (single-shot numbers, ~±25% node contention
noise — trust same-run ratios):

| config | MiB/s |
|---|---|
| `pmd_hdf5`, openPMD defaults | 311 |
| `pmd_hdf5`, rank-aligned chunks | 625 |
| `pmd_adios2` (per-writer files) | 1069 |
| raw `hdf5` backend | 1129 |
| **`pmd_hdf5`, rank chunks + `lfs setstripe -c 8` dir** | **1200** |

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
* `pmd_*` write path is the canonical openPMD pattern (mirrors
  examples/8a + 3b at 0.17.1): owning `shared_ptr` `storeChunk`, scalar
  records used directly on the Record (`BaseRecord` IS-A `RecordComponent`),
  standard `positionOffset` constants, single flush at `iteration.close()`
  (`--pmd-split` opts into the two-flush phase timing).
* `pmd_hdf5` vs `pmd_adios2` gap on Lustre = shared-file coordination tax,
  attributed row-by-row by `scripts/pmd_env_matrix.sh`: chunk layout is
  ~2x (fixed by rank-aligned chunks), file locking through the MDS ~19%,
  transfer mode / paged file-space / deferred metadata ~0.  The dominant
  factor is the filesystem itself: one file = one OST, so the striped
  output directory (table above) is what puts HDF5 on top.  `pmd_adios2`
  fans out across OSTs implicitly (per-writer files).

## Integration (mpiio_evolve)

The tuning harness injects candidates via `ROMIO_HINTS` (and
`lfs setstripe` on the data directory), so pio-bench must not pre-set
MPI_Info keys: with no `--aggregators/--buffer/--no-cb` flags its Info is
empty and the environment's hints take full effect. The JSONL output is
the measurement feed; built inside the compute container image, pinned
by release tag.

External hint channel: `PIOB_ROMIO_HINTS="cb_nodes=4;romio_cb_write=disable"`
(pairs `;`- or `:`-separated) is injected into the file-open `MPI_Info` on
top of the environment, before CLI flags (which always win per key). This
is the portable surface — app-side MPI_Info works on every MPI/ROMIO
build, whereas the ROMIO_HINTS *file* was removed in ROMIO 3.2 and
`MPI_Info_env` is an MPICH-family convention. The banner always echoes the
injected value (`pio_hints="..."`) so no run hides its configuration.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
