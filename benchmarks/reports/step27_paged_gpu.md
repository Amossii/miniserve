# MiniServe Benchmark Report

Raw data: `benchmarks/results/step27_paged_gpu.json`

## Experiment contract

- Model: `/home/henry/project/models/Qwen2.5-0.5B-Instruct`
- Device: `cuda` / `NVIDIA GeForce RTX 4070 Laptop GPU`
- dtype: `torch.bfloat16`
- Arrival: `burst`, request rate: `50` requests/s
- Requests: `12`, max new tokens: `8`, seed: `27`

## Results

| Policy | Capacity | Token budget | Repeats | Output tok/s median [min, max] | TTFT P50 / P99 (ms) | ITL P50 / P99 (ms) | Peak allocated (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| continuous | 4 | 128 | 3 | 70.65 [69.18, 70.72] | 257.29 / 517.74 | 52.56 / 102.87 | 981.5 |

## Interpretation guardrails

- Latency percentiles are recomputed from pooled raw samples; per-run P99 values are not averaged.
- Throughput uses the median of complete-workload observations and keeps the observed range visible.
- A low offered request rate can cap measured throughput before the engine reaches saturation.
- Peak allocated memory includes model residency because the model remains loaded during measurement.
- This synchronous in-process benchmark excludes network and tokenizer latency.
