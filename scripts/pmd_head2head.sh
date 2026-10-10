#!/usr/bin/env bash
# Head-to-head: every backend, SAME node, SAME striped directory, SAME
# geometry, back to back in one session.  Cross-run tables are worthless
# on a shared system (~25% contention drift); this script is the fair
# comparison -- whoever wins here wins.
#
# Usage (from any dir):  ./pmd_head2head.sh
# Knobs: BENCH RANKS LOCAL PARTICLES STEPS STRIPE DIR NOLOCK
set -u
BENCH=${BENCH:-/lustre/rz/dbertini2/pio-bench/build/pio_bench}
RANKS=${RANKS:-32}
LOCAL=${LOCAL:-256}
PARTICLES=${PARTICLES:-4000000}
STEPS=${STEPS:-3}          # mean over 3 checkpoints, not a cold single shot
STRIPE=${STRIPE:-8}
DIR=${DIR:-./h2h}
NOLOCK=${NOLOCK:-FALSE}    # HDF5 file locking; FALSE = the tuned default here

unset OPENPMD_HDF5_PAGED_ALLOCATION OPENPMD_HDF5_DEFER_METADATA \
      OPENPMD_HDF5_INDEPENDENT HDF5_USE_FILE_LOCKING PIOB_PMD_OPTS

rm -rf "$DIR" && mkdir -p "$DIR"
if command -v lfs >/dev/null 2>&1; then
    lfs setstripe -c "$STRIPE" "$DIR"
    echo "output dir: $DIR  ($(lfs getstripe "$DIR" 2>/dev/null | head -1))"
else
    echo "output dir: $DIR  (no lfs: UNSTRIPED -- single-OST ceiling!)"
fi
echo "ranks=$RANKS local=$LOCAL particles=$PARTICLES steps=$STEPS locking=$NOLOCK"
echo

for be in mpiio hdf5 pmd_hdf5 adios2 pmd_adios2; do
    echo "==== $be"
    HDF5_USE_FILE_LOCKING=$NOLOCK \
    mpirun -n "$RANKS" "$BENCH" --backend "$be" --dir "$DIR" \
        --local "$LOCAL" --particles "$PARTICLES" --steps "$STEPS" \
        > /tmp/h2h_$be.log 2>&1
    rc=$?
    grep -E "write MiB/s|wall s/checkpoint" /tmp/h2h_$be.log
    [ $rc -ne 0 ] && { echo "exit=$rc"; tail -4 /tmp/h2h_$be.log; }
    echo
done
echo "done (per-backend logs: /tmp/h2h_<backend>.log)"
