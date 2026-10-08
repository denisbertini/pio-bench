#!/usr/bin/env bash
# pio-bench smoke test: exercise every backend with read-back verification.
#
# Usage (typically inside the compute container):
#   scripts/smoke.sh
#
# Tunables via environment:
#   BIN=build/pio_bench          benchmark binary
#   MPIRUN=mpirun                launcher (e.g. "mpiexec --oversubscribe")
#   RANKS=4  LOCAL=64  PARTICLES=100000  STEPS=3   run geometry
#
# Exit code 0 if every AVAILABLE backend verifies; backends compiled out
# of this build are reported SKIPPED (not failure).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

BIN="${BIN:-$REPO_DIR/build/pio_bench}"
MPIRUN="${MPIRUN:-mpirun}"
RANKS="${RANKS:-4}"
LOCAL="${LOCAL:-64}"
PARTICLES="${PARTICLES:-100000}"
STEPS="${STEPS:-3}"

[ -x "$BIN" ] || { echo "ERROR: benchmark binary not found: $BIN (build first)"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pio_smoke.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "pio-bench smoke: bin=$BIN ranks=$RANKS local=$LOCAL particles=$PARTICLES steps=$STEPS"
echo

fail=0
ran=0
for b in mpiio hdf5 adios2; do
    echo "=== backend: $b ==="
    ( cd "$WORK" && $MPIRUN -n "$RANKS" "$BIN" \
        --backend "$b" --local "$LOCAL" --particles "$PARTICLES" \
        --steps "$STEPS" --interval 1 --verify ) 2>&1 | tee "$WORK/$b.log"
    rc=${PIPESTATUS[0]}
    if grep -q "SKIPPED:" "$WORK/$b.log"; then
        echo "-- $b: SKIPPED (not compiled into this build)"
    else
        case $rc in
            0) echo "-- $b: PASS"; ran=$((ran+1));;
            2) echo "-- $b: SKIPPED (not compiled into this build)";;
            *) echo "-- $b: FAIL (rc=$rc)"; fail=1;;
        esac
    fi
    echo
done

if [ -s "$WORK/pio_metrics.jsonl" ]; then
    echo "metrics collected (example line):"
    head -n1 "$WORK/pio_metrics.jsonl"
fi

if [ "$ran" -eq 0 ]; then
    echo "WARNING: no backend was actually exercised"
    exit 1
fi
[ "$fail" -eq 0 ] && echo "smoke: OK ($ran backend(s) verified)" || echo "smoke: FAILURES"
exit $fail
