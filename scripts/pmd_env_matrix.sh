#!/usr/bin/env bash
# pmd_hdf5 post-fix env matrix: baseline + each openPMD default OFF, one at a
# time (variables are read ONCE by openPMD's handler ctor, hence a fresh
# mpirun per row). The h5probe ignores these vars -- it must run on pio_bench.
set -u
BENCH=${BENCH:-/lustre/rz/dbertini2/pio-bench/build/pio_bench}
RANKS=${RANKS:-32}
run() {
    echo "==== $1"
    mpirun -n "$RANKS" "$BENCH" --backend pmd_hdf5 --dir=./ --local 256 \
           --particles 4000000 --steps 1 --keep-files 2>&1 \
      | grep -E "chkpt |Failed to write" | head -3
}
unset OPENPMD_HDF5_PAGED_ALLOCATION OPENPMD_HDF5_DEFER_METADATA OPENPMD_HDF5_INDEPENDENT
run "baseline (all openPMD defaults ON) -- expect death at mass"
export OPENPMD_HDF5_PAGED_ALLOCATION=OFF; run "PAGED_ALLOCATION=OFF"
unset   OPENPMD_HDF5_PAGED_ALLOCATION
export OPENPMD_HDF5_DEFER_METADATA=OFF;   run "DEFER_METADATA=OFF"
unset   OPENPMD_HDF5_DEFER_METADATA
export OPENPMD_HDF5_INDEPENDENT=OFF;      run "INDEPENDENT=OFF (collective data)"
export OPENPMD_HDF5_PAGED_ALLOCATION=OFF OPENPMD_HDF5_DEFER_METADATA=OFF OPENPMD_HDF5_INDEPENDENT=OFF
run "all three OFF"
unset OPENPMD_HDF5_PAGED_ALLOCATION OPENPMD_HDF5_DEFER_METADATA OPENPMD_HDF5_INDEPENDENT
