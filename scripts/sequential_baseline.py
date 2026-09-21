import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.sequential import SequentialRunner

MODEL_ID = "/home/henry/project/models/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 32


# --------------------------------------------------
# 将一段用户文本转换成 Request。
#
# Tokenizer 仍然属于 workload / serving ingress 层，
# Request 本身只保存 token IDs。
# --------------------------------------------------
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

    prompt_token_ids = encoded["input_ids"][0].tolist()

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
    )


# --------------------------------------------------
# 运行真实 Qwen Sequential Serving baseline。
#
# 这里只做 sanity measurement，
# Step 19～21 才会正式做 serving benchmark。
# --------------------------------------------------
def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)

    model.eval()

    requests = [
        build_request(
            tokenizer,
            request_id="req-A",
            text=("Explain KV cache in LLM inference in one short paragraph."),
            max_new_tokens=MAX_NEW_TOKENS,
        ),
        build_request(
            tokenizer,
            request_id="req-B",
            text=("Explain continuous batching in one short paragraph."),
            max_new_tokens=MAX_NEW_TOKENS,
        ),
        build_request(
            tokenizer,
            request_id="req-C",
            text=("Explain what TTFT means in LLM serving."),
            max_new_tokens=MAX_NEW_TOKENS,
        ),
    ]

    engine = Engine(
        model=model,
        device=device,
    )

    runner = SequentialRunner(engine)

    torch.cuda.synchronize()

    start = time.perf_counter()

    finished_requests = runner.run_requests(requests)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    total_output_tokens = sum(
        request.num_generated_tokens for request in finished_requests
    )

    print("=== Sequential Baseline ===")

    print(f"Requests: {len(finished_requests)}")

    print(f"Elapsed: {elapsed:.3f} s")

    print(f"Output tokens: {total_output_tokens}")

    print(f"Aggregate output throughput: {total_output_tokens / elapsed:.2f} tok/s")

    print()

    for request in finished_requests:
        text = tokenizer.decode(
            request.generated_token_ids,
            skip_special_tokens=True,
        )

        print(f"[{request.request_id}]")

        print(f"prompt tokens: {request.prompt_length}")

        print(f"output tokens: {request.num_generated_tokens}")

        print(f"TTFT: {request.ttft_seconds * 1000:.2f} ms")

        print(f"E2E: {request.e2e_latency_seconds * 1000:.2f} ms")

        print(text)
        print()


if __name__ == "__main__":
    main()
