import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.sequential import SequentialRunner
from miniserve.static_batch import StaticBatchRunner

MODEL_ID = "/home/henry/project/models/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 32


# 将文本 tokenizer 成 MiniServe Request。
def build_request(
    tokenizer,
    *,
    request_id: str,
    text: str,
    max_new_tokens: int,
) -> Request:

    messages = [
        {
            "role": "user",
            "content": text,
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    return Request(
        request_id=request_id,
        prompt_token_ids=(encoded["input_ids"][0].tolist()),
        max_new_tokens=max_new_tokens,
    )


# 根据相同文本重新创建一组 Request。
#
# Request 是有状态对象；
# Sequential 跑完后不能直接拿同一个对象再跑 Static。
def build_workload(
    tokenizer,
) -> list[Request]:

    prompts = [
        (
            "req-A",
            "Explain KV cache in one short paragraph.",
        ),
        (
            "req-B",
            (
                "Explain continuous batching in LLM "
                "serving and why it improves GPU "
                "utilization in one short paragraph."
            ),
        ),
        (
            "req-C",
            "What does TTFT mean?",
        ),
    ]

    return [
        build_request(
            tokenizer,
            request_id=request_id,
            text=text,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        for request_id, text in prompts
    ]


# 打印一组 Request 的生成结果。
def print_results(
    title: str,
    tokenizer,
    requests: list[Request],
) -> None:

    print()
    print(title)

    for request in requests:
        text = tokenizer.decode(
            request.generated_token_ids,
            skip_special_tokens=True,
        )

        print(
            f"{request.request_id}: "
            f"prompt={request.prompt_length}, "
            f"output={request.num_generated_tokens}"
        )

        print(text)
        print()


# 主实验：
#
# 使用完全相同的模型、greedy decoding 和 workload，
# 比较 Sequential 与 Static Batch。
def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Decoder-only batched generation 使用 left padding。
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither PAD nor EOS token.")

        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)

    model.eval()

    eos_token_ids_raw = model.generation_config.eos_token_id

    if eos_token_ids_raw is None:
        eos_token_ids: set[int] = set()

    elif isinstance(
        eos_token_ids_raw,
        int,
    ):
        eos_token_ids = {eos_token_ids_raw}

    else:
        eos_token_ids = set(eos_token_ids_raw)

    # ==================================================
    # Sequential
    # ==================================================

    sequential_requests = build_workload(tokenizer)

    sequential_engine = Engine(
        model=model,
        device=device,
    )

    sequential_runner = SequentialRunner(sequential_engine)

    # 简单 warmup。
    warmup_request = build_request(
        tokenizer,
        request_id="warmup",
        text="Hello.",
        max_new_tokens=8,
    )

    sequential_runner.run_request(warmup_request)

    torch.cuda.synchronize()

    start = time.perf_counter()

    sequential_runner.run_requests(sequential_requests)

    torch.cuda.synchronize()

    sequential_elapsed = time.perf_counter() - start

    # ==================================================
    # Static Batch
    # ==================================================

    static_requests = build_workload(tokenizer)

    static_runner = StaticBatchRunner(
        model=model,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_ids=eos_token_ids,
    )

    # Static 路径自己的 warmup workload。
    static_warmup = [
        build_request(
            tokenizer,
            request_id="warmup-A",
            text="Hello.",
            max_new_tokens=8,
        ),
        build_request(
            tokenizer,
            request_id="warmup-B",
            text="Explain GPU in one sentence.",
            max_new_tokens=8,
        ),
    ]

    static_runner.run_requests(static_warmup)

    torch.cuda.synchronize()

    start = time.perf_counter()

    trace = static_runner.run_requests(static_requests)

    torch.cuda.synchronize()

    static_elapsed = time.perf_counter() - start

    # ==================================================
    # Correctness
    # ==================================================

    print("=== Token-level Correctness ===")

    for sequential, static in zip(
        sequential_requests,
        static_requests,
        strict=True,
    ):
        identical = sequential.generated_token_ids == static.generated_token_ids

        print(f"{sequential.request_id}: {identical}")

        # if not identical:
        #     raise AssertionError(
        #         f"{sequential.request_id} differs "
        #         "between sequential and static batching."
        #     )

    # ==================================================
    # Performance sanity comparison
    # ==================================================

    sequential_tokens = sum(
        request.num_generated_tokens for request in sequential_requests
    )

    static_tokens = sum(request.num_generated_tokens for request in static_requests)

    print()
    print("=== Sanity Performance ===")

    print(
        f"Sequential: "
        f"{sequential_elapsed:.3f}s, "
        f"{sequential_tokens / sequential_elapsed:.2f} tok/s"
    )

    print(
        f"Static:     {static_elapsed:.3f}s, {static_tokens / static_elapsed:.2f} tok/s"
    )

    print()
    print("=== Static Batch Trace ===")

    print(
        "Forward shapes:",
        trace.forward_input_shapes,
    )

    print(
        "Cache lengths:",
        trace.cache_lengths,
    )

    print(
        "Active counts:",
        trace.active_counts,
    )

    print_results(
        "=== Sequential Outputs ===",
        tokenizer,
        sequential_requests,
    )

    print_results(
        "=== Static Outputs ===",
        tokenizer,
        static_requests,
    )


if __name__ == "__main__":
    main()
