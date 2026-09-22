# MiniServe Benchmark Report

Raw data: `benchmarks/results/step22_qwen_gpu_burst.json`

## Experiment contract

- Model: `/home/henry/project/models/Qwen2.5-0.5B-Instruct`
- Device: `cuda` / `NVIDIA GeForce RTX 4070 Laptop GPU`
- dtype: `torch.bfloat16`
- Arrival: `burst`, request rate: `50` requests/s
- Requests: `12`, max new tokens: `8`, seed: `22`

## Results

| Policy | Capacity | Token budget | Repeats | Output tok/s median [min, max] | TTFT P50 / P99 (ms) | ITL P50 / P99 (ms) | Peak allocated (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| continuous | 2 | 64 | 3 | 28.91 [26.79, 31.20] | 769.57 / 1693.66 | 55.82 / 151.01 | 970.0 |
| continuous | 2 | 128 | 3 | 29.20 [27.88, 30.15] | 692.93 / 1629.42 | 63.92 / 136.40 | 970.0 |
| continuous | 4 | 64 | 3 | 37.92 [36.65, 42.68] | 514.85 / 1112.74 | 86.77 / 174.76 | 970.4 |
| continuous | 4 | 128 | 3 | 45.99 [27.06, 48.80] | 457.08 / 1266.62 | 86.55 / 293.65 | 970.9 |
| sequential | 1 | 64 | 3 | 20.13 [18.81, 23.51] | 1280.92 / 2582.11 | 41.79 / 76.27 | 969.5 |
| sequential | 1 | 128 | 3 | 20.11 [15.40, 23.82] | 1122.91 / 3064.97 | 47.33 / 98.67 | 969.5 |

## Interpretation guardrails

- Latency percentiles are recomputed from pooled raw samples; per-run P99 values are not averaged.
- Throughput uses the median of complete-workload observations and keeps the observed range visible.
- A low offered request rate can cap measured throughput before the engine reaches saturation.
- Peak allocated memory includes model residency because the model remains loaded during measurement.
- This synchronous in-process benchmark excludes network and tokenizer latency.
