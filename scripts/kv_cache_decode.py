from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
)

from miniserve.benchmark import benchmark_wall_clock

MODEL_ID = "/home/henry/project/models/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 128


@dataclass
class DecodeResult:
    """
    保存一次生成的结果以及用于观察推理行为的 trace。

    output_ids:
        prompt + generated tokens

    forward_input_lengths:
        每一轮真正送入 model.forward() 的 token 数量。

        无 KV Cache 时应该类似：
        [prompt_len, prompt_len + 1, prompt_len + 2, ...]

        有 KV Cache 时应该类似：
        [prompt_len, 1, 1, 1, ...]

    cache_lengths:
        每一轮 forward 之后 KV Cache 中已经保存的 token 数。
        无 KV Cache 的实现中为空。
    """

    output_ids: torch.Tensor
    forward_input_lengths: list[int]
    cache_lengths: list[int]


# 将模型配置中的 eos_token_id 统一转换成 set。
# 不同模型可能只有一个 EOS，也可能配置多个终止 token。
def normalize_eos_token_ids(
    eos_token_id: int | list[int] | None,
) -> set[int]:

    if eos_token_id is None:
        return set()

    if isinstance(eos_token_id, int):
        return {eos_token_id}

    return set(eos_token_id)


# Step 5 的 naive autoregressive decoding。
#
# 每生成一个 token，都把完整历史 sequence 再送入模型。
# 它是本节的 correctness/performance baseline。
@torch.inference_mode()
def generate_without_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> DecodeResult:

    eos_token_ids = normalize_eos_token_ids(model.generation_config.eos_token_id)

    current_input_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone()

    forward_input_lengths: list[int] = []

    for _ in range(max_new_tokens):
        # 记录本轮真正进入 model.forward() 的 token 数。
        #
        # 无 cache：
        # prompt_len
        # prompt_len + 1
        # prompt_len + 2
        # ...
        forward_input_lengths.append(current_input_ids.shape[1])

        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            use_cache=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        current_input_ids = torch.cat(
            [current_input_ids, next_token],
            dim=1,
        )

        # 如果模型已经生成 EOS，这个 request 完成。
        if next_token.item() in eos_token_ids:
            break

        # 新 token 是有效 token，因此 attention mask 增加一个 1。
        new_mask = torch.ones(
            (current_attention_mask.shape[0], 1),
            dtype=current_attention_mask.dtype,
            device=current_attention_mask.device,
        )

        current_attention_mask = torch.cat(
            [current_attention_mask, new_mask],
            dim=1,
        )

    return DecodeResult(
        output_ids=current_input_ids,
        forward_input_lengths=forward_input_lengths,
        cache_lengths=[],
    )


# 使用 DynamicCache 实现真正的 KV-cache autoregressive decoding。
#
# 第一次 forward 是 prefill：
#   完整 prompt -> model
#
# 后续 forward 是 decode：
#   仅 next_token -> model
#
# past_key_values 保存此前所有 token 在每层 attention 中的 K/V。
@torch.inference_mode()
def generate_with_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> DecodeResult:

    eos_token_ids = normalize_eos_token_ids(model.generation_config.eos_token_id)

    # generated_ids 用于保存完整最终 token sequence。
    #
    # 注意：
    # 它和真正送入 model.forward() 的 current_input_ids
    # 从第二轮开始不再是同一个东西。
    generated_ids = input_ids.clone()

    # 第一轮是 prefill，因此送入完整 prompt。
    current_input_ids = input_ids

    # attention_mask 始终描述：
    #
    # cached historical tokens
    # +
    # 当前新 token
    #
    # 因此它会随着整个 sequence 增长。
    current_attention_mask = attention_mask.clone()

    # 当前 Transformers 使用 Cache 对象维护 KV。
    #
    # DynamicCache 会随着新 token 的到来动态增长。
    past_key_values = DynamicCache(config=model.config)

    forward_input_lengths: list[int] = []
    cache_lengths: list[int] = []

    for _ in range(max_new_tokens):
        # 核心观察：
        #
        # 第一轮：
        # len(current_input_ids) = prompt_len
        #
        # 后续：
        # len(current_input_ids) = 1
        forward_input_lengths.append(current_input_ids.shape[1])

        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # forward 返回更新后的 Cache。
        past_key_values = outputs.past_key_values

        # 记录第一层 cache 当前已经包含多少个 token。
        cache_lengths.append(past_key_values.get_seq_length())

        next_token_logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        # 完整 sequence 单独保存。
        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=1,
        )

        if next_token.item() in eos_token_ids:
            break

        # ==================================================
        # 关键变化
        # ==================================================
        #
        # 下一次 forward 不再传：
        #
        # [prompt + generated history]
        #
        # 而只传刚生成的：
        #
        # [next_token]
        #
        # 过去 token 的 K/V 已经存在 past_key_values 中。
        current_input_ids = next_token

        # attention_mask 不能只有当前 token。
        #
        # KV Cache 中虽然保存了历史信息，但 attention 仍然需要知道
        # 历史 token + 当前 token 哪些位置有效。
        new_mask = torch.ones(
            (current_attention_mask.shape[0], 1),
            dtype=current_attention_mask.dtype,
            device=current_attention_mask.device,
        )

        current_attention_mask = torch.cat(
            [current_attention_mask, new_mask],
            dim=1,
        )

    return DecodeResult(
        output_ids=generated_ids,
        forward_input_lengths=forward_input_lengths,
        cache_lengths=cache_lengths,
    )


# 将完整生成结果去掉 prompt，只 decode 模型真正新生成的 token。
def decode_generated_text(
    tokenizer,
    output_ids: torch.Tensor,
    prompt_length: int,
) -> str:

    generated_ids = output_ids[:, prompt_length:]

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )


# 打印模型配置对应的理论 KV Cache 大小。
#
# 对普通 decoder-only Transformer：
#
# 每个 token 的 KV bytes ≈
#
# 2
# × num_layers
# × num_kv_heads
# × head_dim
# × bytes_per_element
#
# 前面的 2 分别代表 Key 和 Value。
def print_kv_cache_estimate(
    model,
    dtype: torch.dtype,
) -> None:

    config = model.config

    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads

    # GQA/MQA 模型的 KV head 数可能小于 query head 数。
    num_kv_heads = getattr(
        config,
        "num_key_value_heads",
        num_attention_heads,
    )

    head_dim = config.hidden_size // num_attention_heads

    bytes_per_element = torch.empty(
        (),
        dtype=dtype,
    ).element_size()

    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element

    print("=== Theoretical KV Cache ===")
    print(f"Layers:          {num_layers}")
    print(f"Attention heads: {num_attention_heads}")
    print(f"KV heads:        {num_kv_heads}")
    print(f"Head dim:        {head_dim}")
    print(f"KV/token:        {bytes_per_token / 1024:.2f} KiB")
    print()


# 主程序：
#
# 1. 加载模型
# 2. 分别运行 no-cache 与 cache 两条路径
# 3. 检查 token-level correctness
# 4. 查看每轮 forward 的输入长度
# 5. 做一个简单 latency 对比
def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    device = torch.device("cuda")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)

    model.eval()

    print()
    print_kv_cache_estimate(
        model=model,
        dtype=dtype,
    )

    messages = [
        {
            "role": "user",
            "content": ("你好，你会做什么？"),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    prompt_length = input_ids.shape[1]

    print("=== Input ===")
    print(f"Prompt tokens: {prompt_length}")
    print()

    # ==================================================
    # 1. No KV Cache
    # ==================================================

    no_cache_result = generate_without_cache(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # ==================================================
    # 2. With KV Cache
    # ==================================================

    cache_result = generate_with_cache(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # ==================================================
    # 3. Correctness
    # ==================================================

    identical = torch.equal(
        no_cache_result.output_ids,
        cache_result.output_ids,
    )

    print("=== Correctness ===")
    print(f"Token IDs identical: {identical}")
    print()
    print(
        decode_generated_text(
            tokenizer=tokenizer,
            output_ids=no_cache_result.output_ids,
            prompt_length=prompt_length,
        )
    )
    print(
        decode_generated_text(
            tokenizer=tokenizer,
            output_ids=cache_result.output_ids,
            prompt_length=prompt_length,
        )
    )

    # if not identical:
    #     raise AssertionError("KV-cache decoding differs from the no-cache reference.")

    text = decode_generated_text(
        tokenizer=tokenizer,
        output_ids=cache_result.output_ids,
        prompt_length=prompt_length,
    )

    print("=== Generated Text ===")
    print(text)
    print()

    # ==================================================
    # 4. 查看两种实现真正 forward 了多少 token
    # ==================================================

    print("=== Forward Input Length Trace ===")

    print(
        "No cache:",
        no_cache_result.forward_input_lengths,
    )

    print(
        "KV cache:",
        cache_result.forward_input_lengths,
    )

    print(
        "Cache length:",
        cache_result.cache_lengths,
    )

    print()

    # ==================================================
    # 5. 简单性能实验
    # ==================================================
    #
    # 这里只做 sanity benchmark。
    # 后面正式 serving benchmark 会使用统一 workload 和更多样本。

    _, no_cache_stats = benchmark_wall_clock(
        lambda: generate_without_cache(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        ),
        warmup=1,
        repeats=5,
    )

    _, cache_stats = benchmark_wall_clock(
        lambda: generate_with_cache(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        ),
        warmup=1,
        repeats=5,
    )

    print("=== Sanity Benchmark ===")

    print(f"No cache mean: {no_cache_stats.mean_ms:.2f} ms")

    print(f"KV cache mean: {cache_stats.mean_ms:.2f} ms")

    speedup = no_cache_stats.mean_ms / cache_stats.mean_ms

    print(f"Observed speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
