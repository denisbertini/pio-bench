#!/bin/bash
#SBATCH --job-name=pio-bench-ci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=pio-bench-ci_%j.out
#SBATCH --error=pio-bench-ci_%j.err
# ci_test.sh -- CI test for the GENERAL parallel I/O stack
# default run:
#   one node, 4 ranks, the three major HPC I/O backends -- mpiio,
#   hdf5 (parallel) and adios2 -- at a minimal size, default file
#   placement (no lfs setstripe), each leg verified by read-back.
#
# Runs the three major HPC I/O backends at a minimal size to test the
# proper installation of the backends and their MPI dependencies. It can
# be used to quickly validate the I/O stack installed on the compute nodes
# of a cluster, or in a container image environment, before trusting it
# for production. For a full multi-node check -- 4 nodes, all five
# backends, real Lustre striping -- use scripts/submit_all.sh.
#
# Exit 0 only if every REQUIRED backend is present and verifies, and no
# exercised backend fails. A backend compiled out of the build is SKIPPED
# (benchmark exit code 2) -- a failure only if it is listed in REQUIRE.
# No --keep-files: each leg's read-back verify happens inside the run, and
# the per-job output dir is removed when the job finishes.
#
# Clone/build paths derive from SLURM_SUBMIT_DIR, so submit from the
# scripts dir (the batch CWD on the compute node is not reliable -- this
# site runs the script from a spool copy):
#   cd scripts && sbatch ci_test.sh
# CONT / BINDPATH are set explicitly below; edit them if the image lives
# elsewhere. REQUIRE="mpiio hdf5 adios2" may be overridden via env to relax
# the gate.
# Scale the run: allocation via sbatch options (they override the #SBATCH
# defaults above), size (per rank) via env, e.g. the submit_all.sh shape:
#   LOCAL=256 PARTICLES=4000000 STEPS=3 \
#     sbatch --nodes=4 --ntasks-per-node=32 --partition=long ci_test.sh
# submitted from the scripts dir (cd scripts && sbatch ci_test.sh), so
# SLURM_SUBMIT_DIR is the scripts dir and the repo root is one level up
A="$SLURM_SUBMIT_DIR"
export CONT=/lustre/rz/dbertini2/mpiio_evolve/container/plasma_pp.sif
# comma-separated (colon inside a spec means src:dest, not a path separator)
export APPTAINER_BINDPATH=/lustre/rz/dbertini2,/lustre/rz/dbertini
export APPTAINER_SHARENS=true
export APPTAINER_CONFIGDIR=/tmp/$USER
export OMPI_MCA_io=romio341
export HDF5_USE_FILE_LOCKING=FALSE

[ -n "${SLURM_SUBMIT_DIR:-}" ] || { echo "E-: SLURM_SUBMIT_DIR unset -- submit from the scripts dir: cd scripts && sbatch ci_test.sh"; exit 1; }
cd "$A" || exit 1
B="$(cd "$A/.." && pwd)/build/pio_bench"
[ -x "$B" ] || { echo "E-: benchmark binary not found: $B (build first)"; exit 1; }

# default striping: no lfs setstripe, files land with the MDT default layout
D="$A/ci_${SLURM_JOB_ID}"
rm -rf "$D"
mkdir -p "$D"

BACKENDS="mpiio hdf5 adios2"   # general I/O backends only (openPMD/pmd_* too specific)
REQUIRE="${REQUIRE:-$BACKENDS}"
LOCAL="${LOCAL:-32}"          # per-rank field matrices (minimal default)
PARTICLES="${PARTICLES:-50000}"
STEPS="${STEPS:-1}"           # checkpoints

O="--dir $D --local $LOCAL --particles $PARTICLES --steps $STEPS --interval 1 --verify"

echo "pio-bench CI: job=$SLURM_JOB_ID ranks=$SLURM_NTASKS local=$LOCAL particles=$PARTICLES steps=$STEPS required=[$REQUIRE] dir=$D"
echo "pio-bench CI: scriptdir=$A binary=$B"
echo

fail=0
ran=0
for b in $BACKENDS; do
    tag=""; for r in $REQUIRE; do [ "$r" = "$b" ] && tag=" [required]"; done

    echo "=== $b$tag ==="
    srun --export=ALL apptainer exec "$CONT" bash -c "$B --backend $b $O"
    rc=$?

    if [ "$rc" -eq 2 ]; then
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

rm -rf "$D"
if [ "$fail" -eq 0 ]; then
    echo "CI: OK -- general I/O stack verified ($ran backend(s) passed)"
else
    echo "CI: FAILED"
fi
exit $fail
