#!/bin/bash
#SBATCH --job-name=pio128
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --partition=long
#SBATCH --output=pio-bench_%j.out
#SBATCH --error=pio-bench_%j.err
# submit_all.sh -- the 128-rank matrix in ONE sbatch job, so every number
# is a within-session comparison:
#   round 1 (default placement; new files land single-OST here) -- all
#     five backends: hdf5, mpiio, pmd_hdf5, adios2, pmd_adios2
#   round 2 (16-stripe directory; files created inside inherit 16
#     stripes) -- the three shared-file backends: mpiio, hdf5, pmd_hdf5.
#     Each leg's stripe-1 reference is its own round-1 leg, same job.
#
# A layout self-check probes one file per directory up front and aborts
# if r1 is not single-OST or r2 is not 16-stripe (the MDT default layout
# can change cluster-wide -- verify, don't assume).
#
# Usage: sbatch scripts/submit_all.sh
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
rm -rf r1 r2          # fresh: stale files could carry a stale layout
mkdir -p r1 r2
B=../build/pio_bench
# --verify runs AFTER the measured window (write stopwatch stops at the
# completion barrier), so it costs job wall time, not benchmark numbers.
O="--local 256 --particles 4000000 --steps 3 --keep-files --verify"
C="--dir ${A}/r1 $O"      # round 1: single-OST placement
C2="--dir ${A}/r2 $O"     # round 2: 16-stripe placement

# round 2: stripe the DIRECTORY -- files created inside inherit the layout
lfs setstripe -c 16 r2

# layout self-check: verify what a data file in each dir ACTUALLY gets
: > r1/.layout_probe
lfs getstripe r1/.layout_probe | grep -qE 'stripe_count: *1$' \
  || { echo "E-: file in r1 is NOT single-OST (MDT default layout changed?) -- aborting"; exit 1; }
rm -f r1/.layout_probe
: > r2/.layout_probe
lfs getstripe r2/.layout_probe | grep -qE 'stripe_count: *16$' \
  || { echo "E-: file in r2 is NOT 16-stripe -- aborting"; exit 1; }
rm -f r2/.layout_probe

# ---- round 1: default placement (single-OST) ----------------
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5     $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend mpiio    $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_hdf5 $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend adios2   $C"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_adios2 $C"

# ---- round 2: 16-stripe files (stripe-1 reference = the same backend's
#      round-1 leg above, same job) ----------------
srun --export=ALL apptainer exec $CONT bash -c "$B --backend mpiio    $C2"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5     $C2"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_hdf5 $C2"

# verify effective layout afterwards:
lfs getstripe r1/*.h5 r1/*.bin r2/*.h5 r2/*.bin | grep -E 'File|stripe_count|obdidx'
