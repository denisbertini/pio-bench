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
#
# Each run gets its OWN output directory: no backend ever H5Fcreate-TRUNCs
# another backend's file. Cross-layout truncation is the fragile path --
# truncating a file whose extents were allocated on demand by a chunked
# dataset, on Lustre/parallel-HDF5, crashed the first 0.5.0 run. Placement
# is unaffected: single-OST files in the same unstriped parent.
#
# --verify runs AFTER the measured window (write stopwatch stops at the
# completion barrier), so it costs job wall time, not benchmark numbers.
#
# Usage: sbatch scripts/submit_full.sh
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

cd /lustre/rz/dbertini2/pio-bench || exit 1
B=build/pio_bench
O="--local 256 --particles 4000000 --steps 3 --keep-files --verify"

# ---- round 1: default placement (files land stripe-1) ----------------
for d in r1_hdf5 r1_h5auto r1_mpiio r1_pmdhdf5 r1_adios2 r1_pmdadios2; do
  rm -rf "$d"; mkdir -p "$d"
done
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5       --dir r1_hdf5      $O"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5       --dir r1_h5auto    $O --h5-auto-chunks"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend mpiio      --dir r1_mpiio     $O"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_hdf5   --dir r1_pmdhdf5   $O"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend adios2     --dir r1_adios2    $O"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_adios2 --dir r1_pmdadios2 $O"

# ---- round 2: pre-create + 16-stripe the shared files ----------------
# (pmd_adios2 has no round-2 entry on purpose: BP5 mints its own
#  per-writer files, placed by the MDS -- pre-creation does not apply)
for d in r2_mpiio r2_hdf5 r2_pmdhdf5; do
  rm -rf "$d"; mkdir -p "$d"
done
for i in 1 2 3; do
  n=$(printf %06d $i)
  : > r2_mpiio/chkpt_${n}_step_${n}.bin;  lfs setstripe -c 16 r2_mpiio/chkpt_${n}_step_${n}.bin
  : > r2_hdf5/chkpt_${n}_step_${n}.h5;    lfs setstripe -c 16 r2_hdf5/chkpt_${n}_step_${n}.h5
  : > r2_pmdhdf5/chkpt_${n}_step_${n}.h5; lfs setstripe -c 16 r2_pmdhdf5/chkpt_${n}_step_${n}.h5
done
srun --export=ALL apptainer exec $CONT bash -c "$B --backend mpiio    --dir r2_mpiio   $O"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend hdf5     --dir r2_hdf5    $O"
srun --export=ALL apptainer exec $CONT bash -c "$B --backend pmd_hdf5 --dir r2_pmdhdf5 $O"

# verify effective layout afterwards (directory stripe != file stripe):
lfs getstripe r1_hdf5/* r1_h5auto/* r1_mpiio/* r1_pmdhdf5/* \
               r2_mpiio/* r2_hdf5/* r2_pmdhdf5/* \
    | grep -E 'File|stripe_count|obdidx'
