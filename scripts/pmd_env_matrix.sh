#!/usr/bin/env bash
# pmd performance matrix (post-fix era).  Variables are read ONCE by
# openPMD's handler ctor / by HDF5 at file-open, hence a fresh mpirun per
# row.  Purpose: attribute the pmd_hdf5 vs pmd_adios2 gap -- the shared-file
# coordination tax (POSIX locks through the Lustre MDS; independent-mode
# chunk allocation) vs the data path itself.
#
# Usage (from a scratch dir):  ./pmd_env_matrix.sh
# Rows are:  NAME  [ENV=VAL ...]  --  [bench flags]
# Knobs:  BENCH=/path/to/pio_bench  RANKS=32  LOCAL=256  PARTICLES=4000000
set -u
BENCH=${BENCH:-/lustre/rz/dbertini2/pio-bench/build/pio_bench}
RANKS=${RANKS:-32}
LOCAL=${LOCAL:-256}
PARTICLES=${PARTICLES:-4000000}

# Hygiene first: a stray exported OPENPMD_*/HDF5_* var silently poisons
# every baseline (this actually happened).  Start from a clean slate and
# let only the rows below set them.
unset OPENPMD_HDF5_PAGED_ALLOCATION OPENPMD_HDF5_DEFER_METADATA \
      OPENPMD_HDF5_INDEPENDENT HDF5_USE_FILE_LOCKING

row() {
    local name=$1; shift
    local envs=() flags=()
    while [ $# -gt 0 ]; do
        if [ "$1" = "--" ]; then shift; flags=("$@"); break; fi
        envs+=("$1"); shift
    done
    echo "==== $name"
    local log=/tmp/pmdrow.log
    env ${envs[@]+"${envs[@]}"} mpirun -n "$RANKS" "$BENCH" \
        --dir=./ --local "$LOCAL" --particles "$PARTICLES" --steps 1 \
        ${flags[@]+"${flags[@]}"} > "$log" 2>&1
    local rc=$?
    grep -E "^chkpt " "$log" | head -1
    if [ $rc -ne 0 ]; then
        echo "exit=$rc; tail:"
        grep -vE "No file ending" "$log" | tail -6
    fi
    echo
}

H="--backend pmd_hdf5"
row "1 baseline: rank chunks, openPMD defaults"      -- $H
row "2 no file locking (MDS round-trips gone)"       HDF5_USE_FILE_LOCKING=FALSE -- $H
row "3 collective data transfers (via MPI)"          OPENPMD_HDF5_INDEPENDENT=OFF -- $H
row "4 no locking + collective transfers"            HDF5_USE_FILE_LOCKING=FALSE OPENPMD_HDF5_INDEPENDENT=OFF -- $H
row "5 paged file-space allocation OFF"              OPENPMD_HDF5_PAGED_ALLOCATION=OFF -- $H
row "6 deferred metadata OFF (eager metadata)"       OPENPMD_HDF5_DEFER_METADATA=OFF -- $H
row "7 library auto chunks (old slow path)"          -- $H --pmd-auto-chunks
row "8 ADIOS2 reference (no coordination)"           -- --backend pmd_adios2

echo "done."
