# Internal implementation details

The non-obvious implementation choices and tuning knobs. For the per-backend
overview and measured performance, see [backends.md](backends.md).

## Data model & metrics

Every checkpoint writes, into **one** file:

* **fields** — one `double` matrix per rank in a global 3-D subarray layout,
  ghost cells excluded from the file;
* **particles** — `{x, y, z, px, py, pz, mass, id}` per particle, generated
  deterministically per rank (seeded by `seed, rank`) and merged into the same
  file across all ranks.

Field values are the flattened global index; particle `id`s are the globally
unique merged index — both cheap to verify independently. `--verify` reads the
checkpoint back (field cells sampled, particle blocks fully) and a mismatch
aborts the job.

Per-phase timings (fields vs particles) and throughput are reported live and
appended as machine-readable JSON Lines to `pio_metrics.jsonl` in the output
directory (`--dir`) — one line per checkpoint, written by rank 0, each stamped
with `benchmark_version`.

## MPI-IO

The `mpiio` backend uses ONE file view per checkpoint (was two): the freshly
opened file's default byte view already addresses the particle records, so they
go in via a collective `MPI_File_write_at_all` at each rank's byte
displacement — no `set_view`; a single `set_view` then switches to the 3-D
field subarray. The subarray datatypes are cached across checkpoints.

Counts use the MPI-4.0 64-bit count API when the stack provides it (probed in
CMake against the actual prototypes — some builds define the `MPI_Count`
typedef while keeping `int` MPI-IO prototypes, which a naive `#ifdef` would
truncate silently); on int-count stacks the per-rank particle count is capped
at 2^31 records (≈128 GiB) with a loud error, not a silent overflow.

## Particle storage

MPI-IO writes particles directly from the in-memory array of structs (MPI
derived record type, zero copies). [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
stores per-member datasets: by default each member is pre-packed into a
contiguous staging vector so `H5Dwrite` takes its fast path; `--h5-strided`
instead hands HDF5 the AoS stride pattern, producing byte-identical files —
the gap between the two quantifies the in-library conversion cost.
[ADIOS2](https://github.com/ornladios/ADIOS2) pre-packs per member into
explicit global-shape variables. With `ghost > 0` the ADIOS2 backend gathers
the field interior into a temporary block (BP5 `Put` requires contiguous data).

## HDF5 chunk layout

The `hdf5` backend chunks its datasets rank-aligned by default since 0.5.0: the
field chunk is the per-rank brick (one contiguous on-disk chunk per rank) and
each particle-column chunk is the min per-rank count (one chunk per rank) — the
`pmd_hdf5` pattern. `scripts/pmd_env_matrix.sh` measured the chunk-layout axis
at ~2x on Lustre: with the old `H5P_DEFAULT` dcpl (contiguous layout) each
hyperslab write scatters its brick into small z-fastest runs inside one data
object — the same pattern the mpiio backend's C-order file pays.

## ADIOS2 / BP5 telemetry

ADIOS2/BP5 computes per-block min/max/sum statistics on every `Put` by default
(StatsLevel 2) — a full extra pass over the payload on the writer CPU, the
leading suspect for BP5's write-rate gap vs. raw MPI-IO. `--no-adi-stats` sets
StatsLevel 0 to isolate that cost; the file then carries no block statistics
(affects later query/transform tools, not reads).

## The `pmd_*` write path

The `pmd_*` write path is the canonical openPMD pattern (mirrors
examples/8a + 3b at 0.17.1): owning `shared_ptr` `storeChunk`, scalar records
used directly on the Record (`BaseRecord` IS-A `RecordComponent`), standard
`positionOffset` constants, single flush at `iteration.close()`
(`--pmd-split` opts into the two-flush phase timing).

The `pmd_hdf5` vs `pmd_adios2` gap on Lustre = shared-file coordination tax,
attributed row-by-row by `scripts/pmd_env_matrix.sh`: chunk layout is ~2x
(fixed by rank-aligned chunks), file locking through the MDS ~19%, transfer
mode / paged file-space / deferred metadata ~0. The dominant factor is the
filesystem itself: data files inherit the output directory's striping, which is
what puts HDF5 on top. `pmd_adios2` fans out across OSTs implicitly
(per-writer files).

## Lustre striping

On Lustre, create the output directory striped — data files created inside
inherit the directory's layout (without it they land single-OST, capping every
backend at one target's rate). The layout policy is MDT-side and can change, so
confirm what a data file actually got:

```sh
mkdir -p chkpts && lfs setstripe -c 8 chkpts/     # 8 OSTs; more for bigger jobs
HDF5_USE_FILE_LOCKING=FALSE mpirun -n 32 ./build/pio_bench \
    --backend pmd_hdf5 --dir ./chkpts ...          # safe: one job, write-once files
lfs getstripe chkpts/* | grep stripe_count         # confirm, don't assume
```

On the Virgo cluster the binary runs inside the compute container.
`scripts/submit_all.sh` is a ready-made `sbatch` job for the 4-node/128-rank
matrix: all five backends in one allocation (round 1 in a plain directory,
round 2 in a 16-stripe one), `--verify` on every leg, and an up-front probe
that aborts if a data file does not land with the intended stripe count.

## Hint injection (mpiio_evolve)

The tuning harness injects candidates via `ROMIO_HINTS` (and `lfs setstripe`
on the data directory), so pio-bench must not pre-set MPI_Info keys: with no
`--aggregators/--buffer/--no-cb` flags its Info is empty and the environment's
hints take full effect. The JSONL output is the measurement feed; built inside
the compute container image, pinned by release tag.

External hint channel: `PIOB_ROMIO_HINTS="cb_nodes=4;romio_cb_write=disable"`
(pairs `;`- or `:`-separated) is injected into the file-open `MPI_Info` on top
of the environment, before CLI flags (which always win per key). This is the
portable surface — app-side MPI_Info works on every MPI/ROMIO build, whereas
the ROMIO_HINTS *file* was removed in ROMIO 3.2 and `MPI_Info_env` is an
MPICH-family convention. The banner always echoes the injected value
(`pio_hints="..."`) so no run hides its configuration.

openPMD escape hatch: `PIOB_PMD_OPTS='{"hdf5":{...}}'` (a JSON object) is
merged into the `pmd_*` Series open options, exposing every openPMD open-time
knob without code changes — e.g. the HDF5 **subfiling VFD**
(`{"hdf5":{"vfd":{"type":"subfiling","ioc_selection":"every_nth_rank",
"stripe_count":-1}}}`). It may not override `backend`; the banner echoes it
(`pmd_opts="..."`).

## Characterizing with Darshan

Under [Darshan](https://github.com/darshan-hpc/darshan):

```sh
DARSHAN_LOGPATH=$PWD darshan-runtime mpirun -n 64 ./build/pio_bench ...
```
