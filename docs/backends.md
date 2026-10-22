# Backends

`pio-bench` writes each checkpoint through one of five backends. They differ in
transport engine, on-disk format, and — what most affects the measured
throughput — how each rank's field data is laid out on disk. For the
implementation choices and tuning knobs, see [internals.md](internals.md).

| Backend      | Transport engine     | File format | Notes |
|--------------|----------------------|-------------|-------|
| `mpiio`      | MPI-IO (ROMIO hints) | raw binary  | Lowest-level reference; particles via a derived record type, field via one subarray view |
| `hdf5`       | Parallel HDF5        | HDF5        | Rank-aligned chunked datasets |
| `adios2`     | ADIOS2 BP5           | BP4/BP5     | Requires `-DPIO_WITH_ADIOS2=ON` |
| `pmd_hdf5`   | openPMD-api → HDF5   | HDF5        | openPMD-compliant layout |
| `pmd_adios2` | openPMD-api → ADIOS2 | BP4/BP5     | openPMD-compliant layout; needs both openPMD-api and ADIOS2 |

## The `pmd_*` backends

The `pmd_*` backends write standard
[openPMD](https://github.com/openPMD/openPMD-standard) files — the checkpoint
format used by production PIC codes such as
[WarpX](https://github.com/BLAST-WarpX/warpx) and
[PIConGPU](https://github.com/ComputationalRadiationPhysics/picongpu): mesh
`field/rho` plus species `electrons` with `position`, `momentum`, `mass` and
`id` columns. The write path follows openPMD's own examples (8a and 3b): data
is handed to [openPMD-api](https://github.com/openPMD/openPMD-api) (0.16 or
newer) as `shared_ptr` buffers and flushed once when the iteration closes.

By default only the total write time is meaningful (openPMD defers the real
work to the final flush); add `--pmd-split` to flush field and particle data
separately and get a phase breakdown.

Chunking: `pmd_chunks=rank` (default) aligns each dataset chunk with the
per-rank box (openPMD's auto-chunker caps at 4 MiB and shatters the brick);
`--pmd-auto-chunks` restores the library default for A/B. The raw `hdf5`
backend uses the same rank-aligned chunking by default.

## Measured performance

Measured on Lustre (Virgo): 128 ranks (4 nodes), 47.6 GiB checkpoint,
single-OST output dir, one job via `scripts/submit_all.sh` (every leg
`--verify`, `HDF5_USE_FILE_LOCKING=FALSE`, mean of 3 checkpoints):

| backend | MiB/s |
|---|---|
| adios2 | 2397 |
| pmd_hdf5 | 2259 |
| pmd_adios2 | 2218 |
| mpiio | 1646 |
| hdf5 | 1579 |

The spread is data layout, not engine speed. `adios2` and the `pmd_*` backends
re-pack each rank's field brick into one contiguous block (and add their own
aggregation); the raw `mpiio`/`hdf5` backends write the brick strided into a
shared C-order object (no re-pack). On Lustre the contiguous-block pattern
beats the strided scatter by ~1.4x — and it is access-bound, not
placement-bound (striping barely moved `mpiio`).

Cross-run tables on a shared system mislead (earlier ones ranked these engines
in nearly the opposite order) — quote only head2head numbers
(`scripts/pmd_head2head.sh`).

## Scripts

```sh
scripts/smoke.sh          # runs every compiled-in backend with --verify
scripts/pmd_env_matrix.sh # pmd throughput attribution matrix (one mpirun
                          # per row: locking, transfer mode, allocation,
                          # metadata, chunking, ADIOS2 reference)
scripts/pmd_head2head.sh  # all five backends back to back in one session --
                          # the number to quote
scripts/submit_all.sh     # SLURM (sbatch): 4 nodes / 128 ranks, the full
                          # matrix in one job
scripts/ci_test.sh        # SLURM (sbatch): 1 node / 4 ranks, the three
                          # general backends at minimal size -- CI gate
```

Backends absent from the build report `SKIPPED`, not failure. Tunables:
`RANKS`, `LOCAL`, `PARTICLES`, `STEPS`, `MPIRUN` (see each script's header);
for the sbatch scripts, `CONT`, `APPTAINER_BINDPATH`, `REQUIRE`, and
`A` / `BIN` (repo scripts dir and benchmark binary, if the paths derived
on the compute node do not resolve to your clone).
