# Step 19：按时间到达的 Workload 与 Serving Benchmark

## 本节目标与系统位置

```text
固定种子生成不可变 RequestSpec
    → 每种策略、每次重复创建全新 Engine / Request
    → 预热（窗口外）
    → 按计划时间到达，驱动 Engine.step()
    → 完全排空
    → 两种时间边界的 Metrics + 原始 JSON
```

`src/miniserve/workload.py` 定义到达计划和驱动；`scripts/benchmark_serving.py` 负责模型初始化、策略矩阵、预热、重复和结果留档。它复用上一课的模型加载与 HF 对照辅助函数，不重写推理执行器。

## 运行

离线 CPU 小模型，比较请求数容量和 token budget：

```bash
.venv/bin/python scripts/benchmark_serving.py \
  --arrival burst --num-requests 24 --max-new-tokens 8 \
  --max-running 1 3 --token-budgets 6 12 --repeats 2 \
  --check-reference --output benchmarks/results/step19_tiny_burst.json
```

本地 Qwen，模拟泊松到达：

```bash
.venv/bin/python scripts/benchmark_serving.py \
  --model /home/henry/project/models/Qwen2.5-0.5B-Instruct \
  --device cpu --arrival poisson --request-rate 5 \
  --num-requests 4 --max-new-tokens 4 \
  --max-running 3 --token-budgets 40 80 --repeats 1 \
  --check-reference --output benchmarks/results/step19_qwen_cpu.json
```

默认运行无需模型下载。GPU 可用时切换 `--device cuda`。默认结果写入 `benchmarks/results/serving.json`，重复执行同一输出路径会覆盖报告，保留实验时应使用不同文件名。

## 到达模型

- `burst`：全部计划在 0 秒到达，`request-rate` 不参与时间计算。
- `constant`：第 i 个请求计划在 `i / request_rate` 秒到达。
- `poisson`：首请求固定在 0，之后间隔采样自指数分布，均值 `1/request_rate`。

请求率是提供负载的参数，不是测出的完成吞吐。有限样本的实际到达间隔不一定精确等于期望均值。

prompt 从现有四种输入中有放回采样，输出上限均匀采样于 `[1, max_new_tokens]`；EOS 可以让实际输出更短。到达和内容使用独立的局部随机流，相同种子下切换到达模式不会改变 prompt 和输出上限。本课尚未实现任意输入长度分布或真实数据集负载，四种短 prompt 不代表生产请求分布。

## 关键设计：不要隐藏同步执行期间的等待

当前 Engine 是同步的。某次 forward 从 0.0 秒执行到 0.1 秒，而 B 计划在 0.05 秒到达，则 B 只能在 0.1 秒提交。

```text
计划到达             实际提交             首 token
0.05                 0.10                 0.20

dispatch lag         = 0.05 s
submitted TTFT       = 0.10 s
scheduled TTFT       = 0.15 s
```

每轮会提交所有已到期请求，不将未来到达重新锚定到实际提交时刻。空闲时 sleep，已有工作时继续推进。睡眠上限 50 ms，不忙等。

报告保留两套指标：

- `submitted_metrics`：Step 18 原有边界，从实际调用 Engine.add_request 开始。
- `scheduled_metrics`：从计划到达开始，把尚不能提交的等待纳入 TTFT、排队和 E2E。
- `dispatch_lag_ms`：单列计划与实际提交的差值，便于识别负载驱动的延迟。

两套 ITL/TPOT 相同；吞吐也相同，均使用从 workload 起点到全部完成的窗口。`scheduled_metrics` 中的 queue wait 包含提交前延迟，不能与仅 Scheduler 内的排队时间混淆。

驱动只在 step 边界接收请求，因此这是一种可解释的进程内开环到达计划实验，不是独立网络客户端或独立线程实时提交压测。计划到达指标也不是实际客户端观测；报告应同时查看 dispatch lag。

## 公平比较与报告

所有策略复用完全相同的规格列表，实际 Request、KV 和队列每次重建。每次测量前做一组独立 burst 预热；模型加载、tokenization、预热、打印、文件写入及 HF 对照均在窗口之外。测量包含时间检查、请求入队和 Python 驱动开销。

`--max-running` 与 `--token-budgets` 构成笛卡尔积。所有预算必须覆盖并发上限及完整 prompt；先校验全部测量配置，再执行。并发 1 是同一 Engine 的串行执行基线，依然遵循相同到达计划。

各重复轮换策略执行顺序，以减轻固定次序偏差，但这不保证消除温度、频率和系统负载干扰。每次重复独立保存，不能平均多个 P99 后声称得到合并 P99。

JSON 包含环境版本、dtype、CPU 线程数、模型标识、种子、全部输入规格、每次指标原始样本与分位数、每请求相对时间戳和输出 token。跨策略/重复输出必须一致，否则失败退出。`--check-reference` 另外验证首份输出与 HF generate 一致；由于其余输出已经逐项比较，覆盖全部已运行配置。

## 验证与分析方法

```bash
.venv/bin/python -m pytest tests/test_workload.py -q
```

假时钟测试覆盖：可复现到达、空闲等待、forward 期间积累多个请求、提交延迟分解、请求状态隔离、坏规格提前拒绝。

分析时依次检查：

1. 输出是否一致、实际输出 token 数是否一致。
2. 请求规格和环境是否可比。
3. dispatch lag 是否明显，是否已经把这部分等待纳入 TTFT。
4. 不同重复之间的波动，尤其是执行工作相同的配置。
5. 然后再讨论吞吐、TTFT 和 ITL 的取舍。

低请求率下吞吐可能受输入负载限制，不能据此推断引擎最大吞吐。小型随机模型和 CPU 测量不代表 GPU 上的大模型性能；少量请求的 P99 不支持稳定尾延迟结论。本课不声称存在某个必然最优的 token budget。

参考：[vLLM bench serve 参数](https://docs.vllm.ai/en/v0.20.0/cli/bench/serve/)。借鉴请求率、种子、重复记录的实验组织方式，并明确 MiniServe 当前同步执行的边界。

## Checkpoint

- 为什么按轮号到达不能公平比较快慢不同的调度策略？
- 为什么提交延迟不能从 TTFT 报告中消失？
- 为什么并发 1 时，预算已经足够容纳单个 prompt，再增大预算通常不会改变执行分组？
- 为什么吞吐接近期望输入速率时，不能直接认定系统已经达到性能上限？

## 本次实际运行记录（CPU）

原始结果：

- [随机小型 Llama / burst](../benchmarks/results/step19_tiny_burst.json)：24 请求，容量 1/3 × 预算 6/12，各重复 2 次。
- [本地 Qwen2.5-0.5B / poisson](../benchmarks/results/step19_qwen_cpu.json)：4 请求，期望到达率 5 requests/s，容量 3，预算 40/80，各一次。

两组实验均通过跨配置/重复输出一致性及 HF reference 检查。

Qwen 本次观测：

| Budget | Output tokens/s | 计划到达 TTFT P50 (ms) | ITL P99 (ms) | Dispatch lag P99 (ms) |
|---:|---:|---:|---:|---:|
| 40 | 4.13 | 863.96 | 787.73 | 513.91 |
| 80 | 3.98 | 889.22 | 845.26 | 519.60 |

这里只能说明脚本已能测出和保留提交延迟，不能由四个请求、单次运行判断预算 40 优于 80。

小模型实验也保留了明显波动：容量 1、预算 12 的两次吞吐约为 435.57 和 1109.83 tokens/s。容量 1 且预算足够容纳单个 prompt 时，预算 6/12 不应改变 batch 成员构造；差异提示运行噪声、预热覆盖或其他环境影响，需要进一步观测，不能归因于 token budget。没有删除这条较慢样本。

这次未执行 GPU 实验，也没有把随机小模型 CPU 吞吐写成模型服务性能结论。
