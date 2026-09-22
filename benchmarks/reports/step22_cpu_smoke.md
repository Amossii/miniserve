# MiniServe Benchmark Report

Raw data: `benchmarks/results/step22_cpu_smoke.json`

## Experiment contract

- Model: `random-tiny-llama`
- Device: `cpu` / `None`
- dtype: `torch.float32`
- Arrival: `burst`, request rate: `50` requests/s
- Requests: `8`, max new tokens: `4`, seed: `22`

## Results

| Policy | Capacity | Token budget | Repeats | Output tok/s median [min, max] | TTFT P50 / P99 (ms) | ITL P50 / P99 (ms) | Peak allocated (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| continuous | 2 | 6 | 2 | 620.39 [480.83, 759.94] | 14.22 / 32.57 | 3.00 / 6.58 | N/A |
| continuous | 2 | 12 | 2 | 650.48 [481.64, 819.31] | 13.76 / 33.37 | 2.87 / 6.94 | N/A |
| continuous | 4 | 6 | 2 | 634.98 [460.18, 809.78] | 10.86 / 34.76 | 3.59 / 7.87 | N/A |
| continuous | 4 | 12 | 2 | 761.45 [591.64, 931.25] | 6.63 / 20.40 | 4.46 / 7.94 | N/A |
| sequential | 1 | 6 | 2 | 478.92 [244.74, 713.11] | 20.41 / 76.05 | 1.87 / 7.08 | N/A |
| sequential | 1 | 12 | 2 | 239.96 [89.15, 390.77] | 36.61 / 212.24 | 2.15 / 145.66 | N/A |

## Interpretation guardrails

- Latency percentiles are recomputed from pooled raw samples; per-run P99 values are not averaged.
- Throughput uses the median of complete-workload observations and keeps the observed range visible.
- A low offered request rate can cap measured throughput before the engine reaches saturation.
- Peak allocated memory includes model residency because the model remains loaded during measurement.
- This synchronous in-process benchmark excludes network and tokenizer latency.
