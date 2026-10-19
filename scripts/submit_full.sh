#!/bin/bash
#SBATCH --job-name=pio128
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --partition=long
#SBATCH --output=pio-bench_%j.out
#SBATCH --error=pio-bench_%j.err
# submit_full.sh -- the 128-rank matrix in ONE sbatch job, so every number
# is a back-to-back (within-session) comparison:
#   round 1 (default placement, data files land single-OST) -- all five
#     backends: hdf5 (rank chunks), hdf5 --h5-auto-chunks (A/B), mpiio,
#     pmd_hdf5, adios2, pmd_adios2
#   round 2 (pre-created files, lfs setstripe -c 16):
#     mpiio | hdf5 | pmd_hdf5
# Usage: sbatch scripts/submit_full.sh
# =============================================
# PIO Test
# =============================================
export CONT=/lustre/rz/dbertini2/mpiio_evolve/container/plasma_pp.sif
export APPTAINER_BINDPATH=/lustre/rz/dbertini2
export APPTAINER_SHARENS=true
export APPTAINER_CONFIGDIR=/tmp/$USER
export OMPI_MCA_io=romio341

cd /lustre/rz/dbertini2/pio-bench || exit 1
mkdir -p r1 r2
B=build/pio_bench
# --verify runs AFTER the measured window (write stopwatch stops at the
# completion barrier), so it costs job wall time, not benchmark numbers.
C="--dir r1 --local 256 --particles 4000000 --steps 3 --keep-files --verify"

# ---- round 1: default placement (files land stripe-1) ----------------
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5     $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5     $C --h5-auto-chunks"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend mpiio    $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_hdf5 $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend adios2   $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_adios2 $C"

# ---- round 2: pre-create + 16-stripe the shared files ----------------
# (pmd_adios2 has no round-2 entry on purpose: BP5 mints its own
#  per-writer files, placed by the MDS -- pre-creation does not apply)
for i in 1 2 3; do
  f="r2/chkpt_$(printf %06d $i)_step_$(printf %06d $i)"
  : > "$f.bin"; lfs setstripe -c 16 "$f.bin"
  : > "$f.h5";  lfs setstripe -c 16 "$f.h5"
done
C2="${C/--dir r1/--dir r2}"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend mpiio    $C2"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5     $C2"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_hdf5 $C2"

# verify effective layout afterwards (directory stripe != file stripe):
lfs getstripe r1/*.h5 r1/*.bin r2/*.h5 r2/*.bin | grep -E 'File|stripe_count|obdidx'
