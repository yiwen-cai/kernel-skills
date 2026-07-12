# Workflow contract

## Status semantics

- `PASS`: correctness, benchmark, baseline gate, environment gate and requested profiler stages passed.
- `REGRESSION`: named-baseline latency exceeded both thresholds on both confirmations.
- `UNSTABLE`: environmental evidence makes the measurement unsuitable for comparison.
- `PROFILE_PARTIAL`: ordinary benchmark data is valid and retained, but a requested profiler stage failed or could not be parsed.
- `INVALID`: hardware, architecture, correctness, configuration or benchmark prerequisites failed.

Return a nonzero process exit code for every status except `PASS`.

## Metric semantics

- Use CUDA Events around the kernel for regression latency. Warm up first and batch short kernels to reduce timer quantization.
- Compute MFU from algorithm FLOPs divided by elapsed time and configured peak compute.
- Compute MBU from minimum algorithm bytes divided by elapsed time and configured peak DRAM bandwidth.
- Compute analytical Roofline ceiling as `min(peak_compute, arithmetic_intensity * peak_bandwidth)`.
- Label MFU, MBU and analytical Roofline as estimates. Use NCU for measured DRAM/SM throughput, cache behavior, occupancy, stalls and instruction evidence.

## Full profiler defaults

- NSYS: CUDA + NVTX trace; use a full-trace fallback if a very short NVTX capture range is not triggered, and disclose the mode.
- NCU: kernel replay, cache control all, clock control base, one configured representative launch.
- Recommended sections: `LaunchStats`, `Occupancy`, `SpeedOfLight`, `MemoryWorkloadAnalysis`, `SchedulerStats`, `WarpStateStats`, `SpeedOfLight_RooflineChart`.
- Parse NCU raw CSV by aligned header/unit/data rows. Never infer a metric value by scanning unrelated cells in a wide row.

## Case hierarchy

- Quick: at least one launch-bound small case and one throughput-bound large case.
- Full: fixed size/shape/dtype/layout tiers chosen for the operator, not an unbounded sweep.
- Representative: one stable, meaningful case selected before profiling; do not choose it after seeing a favorable result.

## Baseline lifecycle

Provide explicit create, update and show operations. Store the baseline name, operation, source run, source SHA, environment identity and case metrics. Reject unsafe names and source paths outside the project's results tree. Never auto-update a baseline during Quick or Full.
