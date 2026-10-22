# Contributing

1. Fork the repository and create a feature branch.
2. Build with all backends enabled to catch integration issues:

   ```sh
   cmake -B build -DCMAKE_CXX_COMPILER=mpicxx \
         -DPIO_WITH_HDF5=ON -DPIO_WITH_ADIOS2=ON
   cmake --build build -j
   ```

3. Run the scripts under `scripts/` (at least `scripts/smoke.sh`) before
   submitting a PR.
4. Open a pull request describing the change and its motivation.

Measured numbers move with the instrument: bump `kBenchmarkVersion` in
`include/pio/benchmark.hpp` whenever a change can alter the numbers the
backends report, and keep the `pio_metrics.jsonl` schema (one line per
checkpoint) machine-readable.

For questions, open a GitHub issue.
