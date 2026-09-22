# Step 23：Phase A 工程整合

## 1. 本节目标

把课程过程中逐步形成的脚本整理成稳定入口，让演示、benchmark 和 profiling 共享同一套模型加载与 Engine 装配逻辑，同时保持 scheduler、prefill、decode 和 KV Cache 的执行语义不变。

## 2. 公共运行时装配

`src/miniserve/runtime.py` 是 CLI 与核心引擎之间的 composition root，集中负责：

- 检查 CPU/CUDA 设备是否在当前进程可用。
- 加载本地 Hugging Face 模型，或创建离线随机 tiny Llama。
- 固定 dtype、attention implementation、seed 和演示 prompts。
- 解析 token budget 默认值并验证完整 prefill 约束。
- 为每次实验创建新的 Scheduler、DecodeBatchRunner 和 Engine。
- 使用独立 Hugging Face greedy generation 做 correctness 对照。

模型可以跨预热和正式运行复用；Engine、Request、队列与 KV 状态不能复用。这一边界既减少模型重复加载，也避免实验状态相互污染。

```text
CLI arguments
     ↓
load_runtime()
  shared model / tokenizer / prompts / device
     ↓
build_engine()
  fresh Scheduler / Runner / Engine / KV lifecycle
     ↓
demo / benchmark / profiler driver
```

核心模块不解析命令行，也不依赖具体模型路径。脚本负责实验策略，`runtime.py` 负责资源装配，Engine 继续只负责协调 scheduler 与 executor。

## 3. 正式入口

| 入口 | 用途 | 结果 |
|---|---|---|
| `scripts/check_env.py` | 检查 PyTorch 和 CUDA 环境 | 终端诊断 |
| `scripts/run_engine.py` | 四请求完整链路与 HF 对照 | iteration trace |
| `scripts/benchmark_serving.py` | 时间驱动策略矩阵 | 原始 JSON |
| `scripts/analyze_benchmark.py` | 聚合原始 benchmark | Markdown 报告 |
| `scripts/profile_engine.py` | Engine phase 时间分解 | phase JSON |
| `scripts/profile_torch.py` | operator/kernel profiling | trace、表格、元数据 |
| `scripts/profile_nsys.sh` | Nsight Systems 启动包装 | `.nsys-rep` |

这些入口都通过包内公共模块工作。脚本之间不再通过 `from run_engine import ...` 相互导入。

## 4. 学习实验与历史脚本

`scripts/manual_decode.py`、`kv_cache_decode.py`、`sequential_baseline.py`、`static_batch_baseline.py` 等文件记录早期课程的局部实现，适合逐步学习对应机制。它们不是当前 continuous batching Engine 的正式运行入口，部分保留了当时的接口与硬编码模型路径。

`scripts/study/` 保存独立 CUDA/profiler 学习实验。它们不参与 MiniServe serving 主路径，也不应作为项目 benchmark 数字来源。

## 5. 运行验证

离线完整演示：

```bash
.venv/bin/python scripts/run_engine.py --max-new-tokens 2 --check-reference
```

离线 benchmark 冒烟：

```bash
.venv/bin/python scripts/benchmark_serving.py \
  --arrival burst --num-requests 4 --max-new-tokens 2 \
  --max-running 1 2 --token-budgets 6 --repeats 1 \
  --check-reference --output /tmp/miniserve-smoke.json
```

项目回归：

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

## 6. Checkpoint

- 为什么模型可以跨实验复用，而 Engine 和 Request 不应该复用？
- 为什么正式脚本不应从另一个脚本导入公共函数？
- composition root 与 Engine 的职责分别是什么？
- 为什么工程重构后仍必须跑 HF reference 和全量测试？

## 7. Git commit

建议提交信息：

```text
refactor: centralize runtime and engine construction
```

## 8. 面试价值

这次重构体现了系统代码的 control plane 边界：配置与资源生命周期由装配层管理，推理核心保持独立。它也保证 benchmark、profiler 与 demo 使用同一执行路径，减少因脚本漂移造成的错误性能结论。
