#!/bin/bash
#SBATCH --job-name=pio-bench-ci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=pio-bench-ci_%j.out
#SBATCH --error=pio-bench-ci_%j.err
# ci_test.sh -- CI gate for the GENERAL parallel I/O stack, as ONE small
# sbatch job (the CI twin of submit_all.sh):
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
# Env overrides (inherited by the sbatch job from the submit shell):
#   CONT=...sif               Apptainer/Singularity image
#   APPTAINER_BINDPATH=...    path bind-mounted into the container
#   REQUIRE="mpiio hdf5 adios2"   backends that MUST be available AND pass
#   A=/path/to/repo/scripts   scripts dir (default: this file's directory;
#                             sbatch runs on a compute node, so set it if the
#                             derived path does not resolve there)
#   BIN=/path/to/pio_bench    benchmark binary (default: <repo>/build/pio_bench)
#   (add --partition=... to the SBATCH header if your cluster needs it)
#
# Usage: sbatch scripts/ci_test.sh
# =============================================
# PIO CI
# =============================================
A="${A:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export CONT="${CONT:-/lustre/rz/dbertini2/mpiio_evolve/container/plasma_pp.sif}"
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-/lustre/rz/dbertini2}"
export APPTAINER_SHARENS=true
export APPTAINER_CONFIGDIR=/tmp/$USER
export OMPI_MCA_io=romio341
export HDF5_USE_FILE_LOCKING=FALSE

cd "$A" || exit 1
B="${BIN:-$(cd "$A/.." && pwd)/build/pio_bench}"    # absolute; set BIN to override
[ -x "$B" ] || { echo "E-: benchmark binary not found: $B (build first, or set BIN)"; exit 1; }

# default striping: no lfs setstripe, files land with the MDT default layout
D="$A/ci_${SLURM_JOB_ID}"
rm -rf "$D"
mkdir -p "$D"
O="--dir $D --local 32 --particles 50000 --steps 1 --interval 1 --verify"

BACKENDS="mpiio hdf5 adios2"   # general I/O backends only (openPMD/pmd_* too specific)
REQUIRE="${REQUIRE:-$BACKENDS}"

echo "pio-bench CI: job=$SLURM_JOB_ID ranks=4 required=[$REQUIRE] dir=$D"
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
