#!/bin/bash
#SBATCH --job-name=pio128r2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --partition=long
#SBATCH --output=pio-bench_r2_%j.out
#SBATCH --error=pio-bench_r2_%j.err
# submit_round2.sh -- round 2 only, SELF-CONTAINED.
#
# The striped-file test needs a stripe-1 CONTROL in the same session
# (cross-session drift on this system is ~25-50%), so each backend runs
# BOTH placements back to back, interleaved per backend:
#   mpiio     : r2ctrl (fresh, lands single-OST) -> r2 (pre-created 16-stripe)
#   hdf5      : same
#   pmd_hdf5  : same
#   adios2    : r2ctrl only (within-session fan-out reference)
#
# r1 from the full-matrix job is left untouched (its round-1 numbers
# remain a cross-session reference).
#
# Usage: sbatch scripts/submit_round2.sh
# =============================================
# PIO Test
# =============================================
export CONT=/lustre/rz/dbertini2/mpiio_evolve/container/plasma_pp.sif
export APPTAINER_BINDPATH=/lustre/rz/dbertini2
export APPTAINER_SHARENS=true
export APPTAINER_CONFIGDIR=/tmp/$USER
export OMPI_MCA_io=romio341
# cluster-tuned default (cf. the 32-rank table): the MDS lock tax is ~19%
export HDF5_USE_FILE_LOCKING=FALSE

A=/lustre/rz/dbertini2/pio-bench/scripts/
cd $A || exit 1
rm -rf r2 r2ctrl
mkdir -p r2 r2ctrl
B=../build/pio_bench
# --verify runs AFTER the measured window (write stopwatch stops at the
# completion barrier), so it costs job wall time, not benchmark numbers.
O="--local 256 --particles 4000000 --steps 3 --keep-files --verify"
C="--dir ${A}/r2ctrl $O"   # control: fresh files, land single-OST
C2="--dir ${A}/r2 $O"      # test: pre-created 16-stripe files

# pre-create + 16-stripe the round-2 files
for i in 1 2 3; do
  f="r2/chkpt_$(printf %06d $i)_step_$(printf %06d $i)"
  rm -f "$f.bin" "$f.h5"  # fresh files: setstripe refuses to re-stripe
  : > "$f.bin"; lfs setstripe -c 16 "$f.bin"
  : > "$f.h5";  lfs setstripe -c 16 "$f.h5"
done

# interleaved per backend: each A/B pair stays adjacent in time
srun --export=ALL apptainer exec $CONT bash -c "$B --backend mpiio    $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend mpiio    $C2"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5     $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5     $C2"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_hdf5 $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_hdf5 $C2"
# within-session fan-out reference (BP5 per-writer files; no round-2 entry)
srun --export=ALL apptainer exec $CONT bash -c "$B --backend adios2   $C"

# verify effective layout afterwards (directory stripe != file stripe):
lfs getstripe r2ctrl/*.h5 r2ctrl/*.bin r2/*.h5 r2/*.bin | grep -E 'File|stripe_count|obdidx'
