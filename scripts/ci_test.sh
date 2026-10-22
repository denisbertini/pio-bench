#!/usr/bin/env bash
# ci_test.sh -- CI gate for the GENERAL parallel I/O stack.
#
# Runs the three major HPC I/O backends -- mpiio, hdf5 (parallel) and adios2 --
# at a minimal size to test the proper installation of the backends and their
# MPI dependencies. It can be used to quickly validate the I/O stack installed
# on the compute nodes of a cluster, or in a container image environment,
# before trusting it for production. For a full multi-node check -- 4 nodes,
# all five backends, real Lustre striping -- use scripts/submit_all.sh.
#
#   exercised : mpiio (MPI-IO/ROMIO)   hdf5 (parallel HDF5)   adios2 (ADIOS2 BP5)
#   not used  : openPMD (pmd_*)  -- too specific for a general I/O gate
#
# Tunables (env):
#   BIN=build/pio_bench      benchmark binary
#   MPIRUN=mpirun            launcher (e.g. "mpiexec --oversubscribe")
#   RANKS=4 LOCAL=32 PARTICLES=50000 STEPS=1   (small by default -> fast)
#   REQUIRE="mpiio hdf5 adios2"   backends that MUST be available AND pass
#
# Exit 0 only if every REQUIRED backend is present and verifies, and no
# exercised backend fails. A backend compiled out of the build is SKIPPED -- a
# failure only if it is listed in REQUIRE.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

BIN="${BIN:-$REPO_DIR/build/pio_bench}"
MPIRUN="${MPIRUN:-mpirun}"
RANKS="${RANKS:-4}"
LOCAL="${LOCAL:-32}"
PARTICLES="${PARTICLES:-50000}"
STEPS="${STEPS:-1}"
REQUIRE="${REQUIRE:-mpiio hdf5 adios2}"

# general I/O backends only (openPMD / pmd_* intentionally excluded)
BACKENDS="mpiio hdf5 adios2"

[ -x "$BIN" ] || { echo "CI ERROR: benchmark binary not found: $BIN (build first)"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pio_ci.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "pio-bench CI: $MPIRUN -n $RANKS   local=$LOCAL particles=$PARTICLES steps=$STEPS"
echo "pio-bench CI: required = [$REQUIRE]"
echo

fail=0
ran=0
for b in $BACKENDS; do
    tag=""; for r in $REQUIRE; do [ "$r" = "$b" ] && tag=" [required]"; done

    echo "=== $b$tag ==="
    ( cd "$WORK" && $MPIRUN -n "$RANKS" "$BIN" \
        --backend "$b" --local "$LOCAL" --particles "$PARTICLES" \
        --steps "$STEPS" --interval 1 --verify ) 2>&1 | tee "$WORK/$b.log"
    rc=${PIPESTATUS[0]}

    if grep -q "SKIPPED:" "$WORK/$b.log" || [ "$rc" -eq 2 ]; then
        if [ -n "$tag" ]; then
            echo "-- $b: SKIPPED (not compiled in) -- but REQUIRED => FAIL"
            fail=1
        else
            echo "-- $b: SKIPPED (not compiled into this build)"
        fi
    elif [ "$rc" -eq 0 ]; then
        echo "-- $b: PASS"
        ran=$((ran + 1))
    else
        echo "-- $b: FAIL (rc=$rc)"
        fail=1
    fi
    echo
done

[ "$ran" -eq 0 ] && { echo "CI ERROR: no backend was actually exercised"; fail=1; }

if [ "$fail" -eq 0 ]; then
    echo "CI: OK -- general I/O stack verified ($ran backend(s) passed)"
else
    echo "CI: FAILED"
fi
exit $fail
