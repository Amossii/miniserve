# MiniServe Benchmark Report

Raw data: `benchmarks/results/step28_chunked_gpu.json`

## Experiment contract

- Model: `/home/henry/project/models/Qwen2.5-0.5B-Instruct`
- Device: `cuda` / `NVIDIA GeForce RTX 4070 Laptop GPU`
- dtype: `torch.bfloat16`
- Arrival: `burst`, request rate: `50` requests/s
- Requests: `12`, max new tokens: `8`, seed: `27`

## Results

| Policy | Capacity | Token budget | Repeats | Output tok/s median [min, max] | TTFT P50 / P99 (ms) | ITL P50 / P99 (ms) | Peak allocated (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| continuous | 4 | 32 | 3 | 47.59 [41.43, 48.60] | 459.02 / 944.17 | 64.55 / 187.92 | 968.4 |

## Interpretation guardrails

- Latency percentiles are recomputed from pooled raw samples; per-run P99 values are not averaged.
- Throughput uses the median of complete-workload observations and keeps the observed range visible.
- A low offered request rate can cap measured throughput before the engine reaches saturation.
- Peak allocated memory includes model residency because the model remains loaded during measurement.
- This synchronous in-process benchmark excludes network and tokenizer latency.
