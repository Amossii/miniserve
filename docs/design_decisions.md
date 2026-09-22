# Phase A Design Decisions

| 决策 | 选择 | 原因 | 代价 / 后续方向 |
|---|---|---|---|
| 调度与执行分离 | Scheduler 只处理元数据，Runner 处理 tensor | 便于独立测试策略和执行正确性 | 高级策略需要扩展 SchedulePlan |
| 一轮成员固定 | 轮首确定 prefill/decode 集合 | 避免新请求同轮产生两个 token | 不能在 forward 中途重新调度 |
| FIFO admission | waiting queue 严格 FIFO | 行为确定、容易验证 | 没有优先级、SLO 或公平策略 |
| 完整 prefill | prompt 一次进入模型 | cache/position 语义简单 | 长 prompt 会阻塞队首；Step 28 做 chunked prefill |
| 每请求 KV | 每个 DecodeState 独立持有 cache | 生命周期清楚，便于 correctness 对照 | 每轮 pack/unpack 和 padding；Step 25–27 改 block/paged KV |
| Greedy decoding | argmax，无采样 | 跨策略和 HF reference 可逐 token 比较 | 尚无 sampling、logits processor |
| 同步进程内驱动 | step 边界提交到期请求 | 指标边界可解释、实现范围可控 | 不包含网络、tokenizer、异步 frontend |
| 两套 TTFT | submitted 与 scheduled 同时报告 | 不隐藏同步 forward 造成的 dispatch lag | scheduled TTFT 仍非真实远端客户端测量 |
| 原始样本留档 | JSON 保存时间戳与 samples | 可重算 P50/P99，避免平均分位数 | 文件体积高于只保存摘要 |
| 公共 runtime | CLI 共用模型/Engine 装配 | demo、benchmark、profiler 走同一路径 | runtime 仍针对单机单模型 |

## 未实现的生产能力

MiniServe Phase A 不包含 HTTP/gRPC frontend、流式返回、取消、超时、优先级、抢占、prefix cache、quantization、tensor parallel、CUDA Graph 或自定义 attention kernel。它也没有生产级 paged KV allocator。

这些边界是有意控制的项目范围：Phase A 证明 autoregressive inference、KV 生命周期、continuous batching、token-budget scheduling、指标与 profiling 的理解；Phase B 再围绕已观测的 KV 数据搬运成本升级内存系统。
