# MiniServe 项目目标、当前状态与 30 步课程路线

更新时间：2026-09-22

## 1. 项目定位

MiniServe 是一个用于系统学习 LLM inference 与 serving 的简化推理引擎：

> MiniServe: A Continuous-Batching LLM Inference Engine

它的目的不是复制 vLLM，也不是只做一个聊天接口。项目通过亲自实现推理循环、KV Cache、请求状态机、调度器、连续批处理、性能指标、负载生成与 profiling，建立申请 LLM Systems / AI Infra / 大模型推理优化实习所需的工程能力。

MiniServe 属于长期学习路线中的 C01 LLM Inference Foundations 与 C05 LLM Serving Systems，并为后续 CUDA、kernel optimization、PyTorch Systems 和 distributed inference 提供实验载体。

## 2. 最终能力目标

项目完成后，应当能够独立解释和演示以下内容。

### 推理机制

- Autoregressive generation、prefill 与 decode。
- `past_key_values`、attention mask、position IDs 与 cache position。
- KV Cache 为什么降低重复计算，以及它为什么成为显存瓶颈。
- 不同 context length 的请求如何共同执行 decode。

### Serving 系统

- Request、Scheduler、Engine、Model Runner 与 Metrics 的职责边界。
- Waiting、Running、Finished 生命周期，以及 Prefill、Decode 执行阶段。
- Admission、continuous batching、dynamic batch membership 与 token budget。
- TTFT、ITL、TPOT、E2E latency、吞吐与分位数的测量边界。

### 性能工程

- 构造可复现 workload，控制请求率、随机种子、并发数和 token budget。
- 使用 profiler 找到 prefill、decode、KV pack/unpack 或 Python 调度中的瓶颈。
- 根据证据选择优化，而不是先写 kernel 再寻找使用场景。
- 以 correctness → benchmark → profiling → optimization → benchmark 的闭环报告结果。

### 工程交付

- 项目可在本地复现，测试、演示、benchmark 和 profiler 入口清楚。
- README、架构图、设计决策和 benchmark 报告足以供陌生 reviewer 阅读。
- GitHub 项目可以写入简历，并能支撑 scheduler、KV Cache、continuous batching 和性能分析相关面试。

## 3. 系统架构

当前主路径：

```text
Request / Workload
        ↓
Scheduler
  waiting / running / admission / token budget
        ↓
Engine.step()
  fixed prefill set + fixed decode set
        ↓
DecodeBatchRunner
  prefill / KV pack / batched decode / KV unpack
        ↓
Hugging Face Causal LM
        ↓
Request timestamps + ServingMetrics
```

主要职责：

| 模块 | 负责 | 不负责 |
|---|---|---|
| Request | Token、生命周期、阶段、时间戳和请求级指标 | 模型 forward、GPU tensor 调度 |
| Scheduler | 队列、回收、FIFO admission、请求数限制、token budget | 模型执行、KV tensor 操作 |
| Engine | 协调调度计划和执行器、维护 request 到 DecodeState 的映射 | 具体 attention/KV tensor 构造 |
| DecodeBatchRunner | Prefill、heterogeneous decode batch、KV pack/unpack | Admission policy、workload 生成 |
| Workload / Benchmark | 到达计划、实验驱动、指标汇总与结果保存 | 改变模型生成语义 |

## 4. 项目边界

Phase A 不要求实现生产级网络服务、完整 PagedAttention、分布式推理或自定义 CUDA kernel。HTTP/流式 API 只有在展示项目时确有价值才考虑，不用它替代核心 inference engine 工作。

Phase B 的高级功能也保持教学规模：先实现可解释的数据结构和调度语义，再讨论性能。任何 GPU 优化必须先有 profiler 证据和 correctness baseline。

## 5. 当前进度

总路线为 30 步。目前完成 **30 / 30**，Phase A 与 Phase B 教学路线均已完成。

| 范围 | 状态 | 说明 |
|---|---|---|
| Step 1–15 | 已完成 | 基础推理、KV、Engine、Scheduler、continuous batching |
| Step 16–19 | 已完成 | Prefill/decode 共存、token budget、metrics、时间驱动 workload |
| Step 20 | 已完成 | Engine phase profiling 与结构化 trace |
| Step 21 | 已完成 | CPU/CUDA operator trace 与 KV pack/unpack 瓶颈证据已保存 |
| Step 22 | 已完成 | RTX 4070 上的 Qwen GPU matrix、峰值显存和聚合报告已保存 |
| Step 23 | 已完成 | 公共 runtime 装配、正式入口边界和回归验证 |
| Step 24 | 已完成 | Architecture、Design Decisions、Phase A Report 与面试材料 |
| Step 25 | 已完成 | 固定大小 logical/physical block 与 metadata allocator |
| Step 26 | 已完成 | Request block table、跨块追加、slot mapping 与 batch metadata |
| Step 27 | 已完成 | Paged KV storage、HF gather adapter、Engine 集成与 GPU A/B |
| Step 28 | 已完成 | Partial prefill state、chunk budget、decode priority 与 GPU 实验 |
| Step 29 | 已完成 | LIFO victim、KV block 回收、队尾重入与 recompute correctness |
| Step 30 | 已完成 | Profiler-driven A/B、失败方案回退与 Phase B 总结 |

当前验证基线：

- 全套测试：118 passed。
- `scripts/run_engine.py` 可运行完整 Engine 链路并与 Hugging Face `generate()` 对照。
- `scripts/benchmark_serving.py` 支持 burst、constant、poisson 到达和多组调度参数比较。
- 已有 CPU smoke、RTX 4070 / Qwen2.5-0.5B GPU benchmark matrix 和 CUDA operator trace。
- `docs/interview_handbook.md` 汇总完整架构、设计取舍、实验结论和面试问答。

“已完成”表示功能、文档和验证完成，不表示已经创建 Git commit。

## 6. 30 步课程路线

### Phase A1：推理基础（Step 1–6，已完成）

#### Step 1：项目 Scope

明确 MiniServe 的目标、非目标、Phase A / Phase B 范围和学习产出。

完成标准：能解释为什么项目重点是 inference engine、scheduler、KV 与 benchmark，而不是聊天 UI。

#### Step 2：开发环境

建立 Python、PyTorch、Transformers、CUDA 与本地模型环境检查。

完成标准：能够确认 Python、依赖、PyTorch build、CUDA availability 和模型路径。

#### Step 3：Hugging Face Baseline

使用 `generate()` 建立 greedy generation correctness 与性能基线。

完成标准：记录模型、dtype、prompt/output 长度、延迟、吞吐与显存口径。

#### Step 4：Benchmark 基础工具

实现墙钟、CUDA event、warmup、重复测量、吞吐与分位数基础工具。

完成标准：能说明 CPU wall clock 与 GPU event 的测量边界。

#### Step 5：手写 Autoregressive Loop

不用 `generate()`，逐 token 调用模型并执行 greedy argmax。

完成标准：输出 token 与 HF baseline 一致，并能解释每轮输入序列如何增长。

#### Step 6：KV Cache

接入 `past_key_values`，正确处理 attention mask、position IDs 与 cache position。

完成标准：输出与无 cache 路径一致；能说明 decode 输入为何从完整序列降为一个 token。

### Phase A2：Engine 与请求模型（Step 7–11，已完成）

#### Step 7：Request Abstraction

将请求 ID、prompt、generated tokens、停止条件和时间信息封装为 Request。

完成标准：请求逻辑状态与模型 tensor/KV 状态分离。

#### Step 8：状态机

实现 WAITING → RUNNING → FINISHED 生命周期，以及 PREFILL → DECODE 阶段转换。

完成标准：非法状态转换明确失败，测试覆盖关键边界。

#### Step 9：Engine Skeleton

形成 `add_request()`、`step()`、`has_unfinished_requests()` 的驱动接口。

完成标准：调用方可以通过循环 `step()` 推进请求，而不直接控制模型内部状态。

#### Step 10：Sequential Baseline

一次只执行一个请求，形成 serving correctness 与性能对照。

完成标准：多请求顺序执行，结果与独立 HF generation 一致。

#### Step 11：Static Batching

让同时到达的请求组成固定 batch，处理 padding、mask 和提前结束行。

完成标准：不同 prompt length 的请求能在固定 batch 中正确生成，并可与 sequential baseline 比较。

### Phase A3：Continuous Batching（Step 12–17，已完成）

#### Step 12：Scheduler Design

定义 waiting、running、finished 集合，明确 control plane 与 execution plane 的边界。

完成标准：形成 scheduler 设计文档和核心 invariants。

#### Step 13：Scheduler V1

实现 FIFO admission、请求数容量、finished 回收和动态 slot 复用。

完成标准：能够手推并测试 `A B C → A C D`。

#### Step 14：Continuous Decode Batching

实现 per-request KV → padding/pack → batched forward → unpack → per-request KV。

完成标准：不同 context length 的请求共同 decode，输出与独立生成一致。

#### Step 15：Dynamic Continuous Batching

让 batch membership 每轮变化，新请求可复用已完成请求的 slot。

完成标准：Engine 能持续接收请求并动态构造 decode batch。

#### Step 16：Prefill 与 Decode 共存

在同一 Engine iteration 中执行新请求 prefill 和已有请求 decode，并在轮首固定两组成员。

完成标准：新请求本轮只产生首 token，下一轮才进入 decode；完成请求轮末回收 KV。

#### Step 17：Token Budget Scheduler

加入 `max_num_batched_tokens`，先为已有工作预留预算，再按严格 FIFO 接纳完整 prefill。

完成标准：每轮输入 token 数不超预算；超长 prompt 明确拒绝；能解释请求数上限与 token budget 的区别。

### Phase A4：Metrics 与 Workload（Step 18–19，已完成）

#### Step 18：请求级 Metrics

记录 token timestamps，计算 queue wait、TTFT、ITL、TPOT、E2E latency 和 throughput。

完成标准：明确进程内事件边界；batched token 先读回 CPU 再计时；边界和公式有确定性测试。

#### Step 19：时间驱动 Workload 与 Serving Benchmark

支持 burst、constant、poisson 到达，固定随机种子，比较并发数与 token budget，并保存原始 JSON。

完成标准：同一 workload 公平复用到不同策略；区分计划到达、实际提交和 dispatch lag；跨配置输出一致并可选 HF 对照。

### Phase A5：Profiling、性能报告与公开交付（Step 20–24）

#### Step 20：Engine Phase Profiling（已完成）

目标：回答一轮 `Engine.step()` 的时间花在哪里。

计划内容：

- 为 scheduler、prefill、KV pack、model forward、KV unpack、状态更新建立分阶段 trace。
- CPU 使用 wall clock；CUDA 路径明确同步边界或使用 CUDA event。
- 将 profiling 与普通 benchmark 分开，避免 instrumentation 改变默认性能路径。
- 输出每轮 batch size、prompt/context length、scheduled tokens 与阶段耗时。

完成标准：产生结构化 trace，并能定位当前实现中最主要的一个或多个耗时阶段。已提供 `EngineProfiler` 和 `scripts/profile_engine.py`；CPU smoke trace 保存在 `benchmarks/results/profile_step20.json`。

#### Step 21：PyTorch Profiler 与 GPU Profiling（进行中）

目标：从 Engine phase 继续下钻到 operator/kernel 层。

计划内容：

- 在有 CUDA 的环境运行 PyTorch Profiler，记录 CPU/CUDA activity、shape 和显存。
- 使用 Nsight Systems 观察 kernel timeline、同步、CPU launch gap 与 GPU utilization。
- 选择关键区间进一步用 Nsight Compute 或等价工具检查 memory/compute bottleneck。
- 将 prefill 与 decode 分开分析，避免用统一结论描述两种工作负载。

完成标准：保存可复现 profiler 命令、trace 文件说明和至少一个有证据的瓶颈结论。已完成 CPU/CUDA PyTorch trace、Engine/runner annotations 和 Nsight 入口；Qwen CUDA trace 中 `aten::cat` 出现 3035 次并报告约 59.24 MiB allocation，与当前 KV pack/unpack 设计相符。完整限制见 `docs/operator_profiling.md`。

#### Step 22：GPU Benchmark Matrix 与性能分析

目标：形成可用于 README 和面试的可信实验，而不是单次跑分。

计划内容：

- 固定模型、GPU、dtype、prompt/output length、到达模式和随机种子。
- 比较 sequential、static batch、continuous batch，以及不同并发数和 token budget。
- 重复实验并报告原始样本、P50/P99、吞吐、TTFT、ITL、峰值显存。
- 解释负载受限与系统饱和的区别，并记录 correctness 与环境元数据。

完成标准：生成 benchmark 表格/图和分析文字，所有结论能够追溯到原始 JSON。已实现 schema v2 峰值 CUDA 显存记录、策略标签和 `scripts/analyze_benchmark.py` 原始样本聚合；RTX 4070 / Qwen2.5-0.5B 的三次重复 burst matrix 已通过 HF reference 和跨策略一致性检查。

#### Step 23：Phase A 工程整合

目标：让仓库成为一个一致、可维护、可复现的工程项目。

计划内容：

- 统一配置、模型加载和 CLI，减少脚本之间的重复与历史接口漂移。
- 整理早期实验脚本，明确 active entry points 与 archived learning experiments。
- 补齐错误信息、资源释放、随机种子、输出目录与结果 schema。
- 只保留有价值的测试，确保 CPU correctness suite 和可选 GPU smoke test 清晰。

完成标准：新用户能按文档完成环境检查、正确性测试、完整演示、benchmark 和 profiling。已新增 `miniserve.runtime` 统一模型加载、设备检查、预算验证、Engine 构造和 HF 对照；正式脚本不再互相导入，入口与历史学习实验的边界记录在 `docs/project_structure.md`。

#### Step 24：Phase A 文档、Benchmark Report 与简历交付

目标：完成简历可投版本并公开展示。

计划内容：

- 更新 README：项目动机、功能、架构图、快速开始、实验结果和限制。
- 完成 Architecture、Design Decisions、Benchmark Report、Profiling Report。
- 记录当前 per-request padded KV 的时间/空间代价，以及与 vLLM/SGLang 的差距。
- 准备简历 bullet、项目介绍和常见面试问题。

完成标准：GitHub reviewer 不依赖课程对话也能理解、运行和评估项目；Phase A tag/release 可创建。已完成 `architecture.md`、`design_decisions.md`、`phase_a_report.md` 和 `interview_guide.md`，README 串联运行入口、实验证据与项目边界。

### Phase B：KV Memory 与高级 Serving（Step 25–30，待完成）

#### Step 25：Block Allocator（已完成）

目标：把 KV 内存管理从“每请求一个动态 tensor”抽象为固定大小 block。

计划内容：定义 logical block、physical block、free list、allocation/free，以及请求结束时的资源回收。

完成标准：allocator invariants 和碎片/耗尽场景有测试；尚不要求直接接入 attention kernel。已实现固定大小 `LogicalBlock`、`PhysicalBlock`、确定性 free heap、原子多块分配/释放、request 级完整回收，以及耗尽、洞复用和非法释放测试。

#### Step 26：Block Table 与 Slot Mapping（已完成）

目标：建立逻辑 sequence position 到物理 KV slot 的映射。

计划内容：每请求 block table、追加 token 时分配、slot mapping、跨 block 边界和 batch metadata。

完成标准：可手推并测试多个请求的逻辑位置、物理 block 与回收过程。已实现 request 级 `BlockTable`、跨 block 原子追加、逻辑 position 到 physical slot 的地址转换、batch row metadata，以及 release 后地址失效和资源复用测试。

#### Step 27：Paged KV Cache Execution（已完成）

目标：让模型执行路径使用 block metadata，减少当前 heterogeneous KV padding/pack/unpack 的浪费。

计划内容：选择适合教学规模的 PyTorch/Triton 实现路径，保持与 HF greedy reference 的 token-level correctness。

完成标准：真实 Engine 走 paged/block KV 路径；测量显存占用和 KV 整理开销，与 Phase A 实现对比。已实现共享预分配 `PagedKVStorage`、paged state、prefill slot 写入、decode gather 与单 token slot write，并接入统一 Engine/CLI。公平预热后的 GPU A/B 中 paged adapter 吞吐中位数下降约 11.3%、峰值 allocated 增加约 10.7 MiB，说明仅分页 storage 而没有直接寻址 kernel 不会自动提速。

#### Step 28：Chunked Prefill（已完成）

目标：让超过单轮预算的长 prompt 跨轮执行，并降低长 prefill 对 decode ITL 的干扰。

计划内容：prefill progress、chunk budget、部分 KV 状态、位置与 mask、decode-priority scheduling。

完成标准：长 prompt 不再因超过预算被拒绝；分块与完整 prefill 输出一致；有 TTFT/ITL trade-off 实验。已实现 Request prefill cursor、`PrefillChunk`、decode-priority 调度、partial DynamicCache 和最终 chunk 首 token 语义；9-token/budget-4 correctness 与 HF 对照通过。短 prompt GPU 实验中 chunked 配置吞吐和延迟均变差，说明 chunk 粒度必须匹配目标 workload。

#### Step 29：Preemption 与 Recompute（已完成）

目标：在 KV/sequence 容量不足时暂停低优先级请求，并通过 recompute 恢复。

计划内容：调度状态、victim policy、KV 释放、重新进入 waiting、starvation 防护和恢复 correctness。

完成标准：构造资源压力 workload，验证无死锁、无 KV 泄漏、请求最终完成，并量化 recompute 代价。已实现显式 block 水位抢占、LIFO victim、waiting 队尾重入、最大抢占次数和 paged cache 重建；压力测试中 victim 重计算 3 个历史 token，最终输出与 Hugging Face `generate()` 一致，结束后所有 blocks 均归还。设计与限制见 `docs/preemption.md`。

#### Step 30：Profiler-Driven Optimization 与 Phase B 总结（已完成）

目标：根据 profiler 证据选择一项真实优化，并完成最终对比。

候选方向：

- 减少 KV metadata/pack 开销。
- `torch.compile` 或算子融合。
- CUDA Graph，前提是 shape 与控制流适合。
- 一个有明确热点证据的 Triton kernel。

完成标准：形成“瓶颈证据 → 方案 → correctness → benchmark → 局限”的闭环；更新最终架构、性能报告、简历描述和下一阶段学习建议。已根据 `aten::cat` trace 实验 direct-copy KV packing；候选输出正确，但 GPU 吞吐中位数下降 20.0%、TTFT/ITL 变差且显存不变，因此回退实现并保留原始 before/after 数据。完整结论见 `docs/phase_b_report.md`。

## 7. 近期执行顺序

30 步既定课程已完成。Phase B 的执行顺序为：

```text
Step 25 Block Allocator（已完成）
    ↓
Step 26 Block Table（已完成）
    ↓
Step 27 Paged KV Execution（已完成）
    ↓
Step 28 Chunked Prefill（已完成）
    ↓
Step 29 Preemption / Recompute（已完成）
    ↓
Step 30 Profiler-Driven Optimization / Phase B 总结（已完成）
```

## 8. 工程原则

每一步遵循：

```text
Understand
    ↓
Implement
    ↓
Test correctness
    ↓
Benchmark
    ↓
Analyze
```

- 不为增加代码量而增加功能。
- 不根据少量 CPU 小模型样本得出 GPU serving 性能结论。
- 不把 token budget 当作精确计算量或显存预算。
- 不平均多个 P99 后称为合并 P99；保留原始样本和实验元数据。
- 任何优化都必须有 profiler 证据、正确性对照和前后 benchmark。
- 遇到错误先按 Python、dependency、PyTorch、CUDA、Transformers API 的顺序定位，不跳过失败。
