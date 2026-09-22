# Step 20：Engine Phase Profiling

Step 20 把一轮 `Engine.step()` 拆成 scheduler、prefill、decode、reclaim 和 total 五个阶段。`EngineProfiler` 默认不启用；只有构造 `Engine(..., profiler=profiler)` 时才创建计时器，因此普通运行路径不承担 profiling 的计时开销。

运行：

```bash
.venv/bin/python scripts/profile_engine.py \
  --output benchmarks/results/profile_step20.json
```

也可以加本地模型：

```bash
.venv/bin/python scripts/profile_engine.py \
  --model /home/henry/project/models/Qwen2.5-0.5B-Instruct \
  --device cpu --max-new-tokens 4 \
  --output benchmarks/results/profile_qwen_cpu.json
```

输出每轮的：

```text
iteration
total_ms
scheduler_ms
prefill_ms
decode_ms
reclaim_ms
num_running / num_prefill / num_decode
prefill_tokens / decode_tokens / scheduled_tokens
```

`prefill_tokens` 是完整 prompt 输入长度，`decode_tokens` 是本轮参与 decode 的请求数；它们是工作量元数据，不是耗时的等价物。decode 阶段仍会读取每个请求的历史 KV，因此 context length 会影响耗时。

## 时间边界

Step 20 是 Engine phase profiling，不是 operator profiler。CPU 上使用 `perf_counter()`；CUDA 上，脚本在 workload 结束后同步设备，但单个 phase 的 GPU kernel 异步语义仍需 Step 21 的 CUDA event 或 PyTorch Profiler 明确处理。

当前 `decode_seconds` 包含 DecodeBatchRunner 的 KV pack、模型 forward、KV unpack 和 Request 更新；它还没有细分这些子阶段。下一课再下钻到 operator/kernel。

不要用这份 trace 直接声称 GPU kernel 性能。它用于回答第一层问题：一轮时间主要花在 prefill、decode 还是 Python control plane。

本次 CPU smoke trace 显示：在这个小型随机 Llama 和四请求 workload 中，decode 通常占 total 的主要部分；这是验证 instrumentation 的观察，不是生产性能结论。
