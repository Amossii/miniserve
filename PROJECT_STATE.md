# MiniServe 当前项目状态


## 当前课程进度

Step 15 已完成。


下一步：

Step 16：

Prefill 与 Decode 共存。


---

# 已完成内容


## Step 1

项目 Scope

完成。


## Step 2

开发环境

完成。


## Step 3

HF baseline

完成。


## Step 4

Benchmark 基础工具

完成。


## Step 5

手写 autoregressive generation loop

完成。


## Step 6

KV Cache

完成。


掌握：

- past_key_values
- position_ids
- attention_mask
- cache_position


## Step 7

Request abstraction

完成。


Request 保存：

- request_id
- prompt_tokens
- generated_tokens
- status
- timestamp


## Step 8

状态机

完成。


状态：

WAITING

↓

RUNNING

↓

FINISHED


阶段：

PREFILL

DECODE


## Step 9

Engine skeleton

完成。


接口：

engine.add_request()

engine.step()

engine.has_unfinished_requests()


## Step 10

Sequential baseline

完成。


## Step 11

Static batching

完成。


## Step 12

Scheduler design

完成。


## Step 13

Scheduler V1

完成：

- waiting queue
- running queue
- admission
- finished removal


## Step 14

Continuous Decode Batching

完成。


支持：

不同长度 request 同时 decode。


## Step 15

Dynamic Continuous Batching

完成。


支持：

Iteration N:

A B C


Iteration N+1:

A C D



---

# 当前限制


- KV Cache 仍然是简化实现
- 没有 token budget
- 没有 block KV cache
- 没有真实 serving API
- benchmark 尚未完善


---

# 下一目标


Step 16：

实现：

Prefill + Decode coexistence


目标：


同一轮：

新请求：

Prefill


旧请求：

Decode


形成更接近 vLLM 的 scheduler execution loop。
