import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from miniserve.decode_batch import (
    DecodeBatchRunner,
    DecodeState,
    normalize_eos_token_ids,
)
from miniserve.request import Request

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 8


# ---------------------------------------------------------
# 将文本转换成 prompt token IDs。
# ---------------------------------------------------------
def tokenize_prompt(
    tokenizer,
    text: str,
) -> list[int]:

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

    return encoded["input_ids"][0].tolist()


# ---------------------------------------------------------
# 创建一组 prompt length 明显不同的 Request。
# ---------------------------------------------------------
def build_requests(
    tokenizer,
) -> list[Request]:

    prompts = [
        (
            "A",
            "What is KV cache?",
        ),
        (
            "B",
            (
                "Explain continuous batching in LLM "
                "serving, including why requests with "
                "different sequence lengths can still "
                "share decode iterations."
            ),
        ),
        (
            "C",
            "Define TTFT.",
        ),
    ]

    requests: list[Request] = []

    for request_id, text in prompts:
        request = Request(
            request_id=request_id,
            prompt_token_ids=tokenize_prompt(
                tokenizer,
                text,
            ),
            max_new_tokens=MAX_NEW_TOKENS,
        )

        # Step 13 之后正常情况下这一步应该由
        # Scheduler admission 完成。
        #
        # 本实验暂时单独测试 execution layer，
        # 所以手动执行 admission。
        request.mark_running()

        requests.append(request)

    return requests


# ---------------------------------------------------------
# 使用 Hugging Face generate() 生成 correctness reference。
#
# 每个 Request 独立执行。
# ---------------------------------------------------------
@torch.inference_mode()
def build_reference(
    model,
    *,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:

    input_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
    )

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        repetition_penalty=1.0,
    )

    generated_ids = output_ids[
        0,
        len(prompt_token_ids) :,
    ]

    return generated_ids.tolist()


# ---------------------------------------------------------
# 主实验：
#
# 1. 三个 Request 独立 prefill。
# 2. 得到不同 logical KV length。
# 3. 多轮 heterogeneous batched decode。
# 4. finished Request 从下一轮 decode set 中移除。
# 5. 与 HF generate() 做完整 token-level correctness。
#
# 注意：
#
# 这里只允许 Request 离开，
# 不允许新的 Request 加入。
#
# A/B/C -> A/C
#
# 可以；
#
# A/B/C -> A/C/D
#
# 留到 Step 15。
# ---------------------------------------------------------
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

    eos_token_ids = normalize_eos_token_ids(model.generation_config.eos_token_id)

    runner = DecodeBatchRunner(
        model=model,
        device=device,
        eos_token_ids=eos_token_ids,
    )

    requests = build_requests(tokenizer)

    # ==================================================
    # 先建立 Reference
    # ==================================================

    references = {
        request.request_id: build_reference(
            model,
            prompt_token_ids=(request.prompt_token_ids),
            max_new_tokens=(request.max_new_tokens),
            device=device,
        )
        for request in requests
    }

    # ==================================================
    # Independent Prefill
    # ==================================================

    states: list[DecodeState] = []

    print("=== PREFILL ===")

    for request in requests:
        state = runner.prefill_request(request)

        print(
            f"{request.request_id}: "
            f"prompt={request.prompt_length}, "
            f"generated="
            f"{request.num_generated_tokens}, "
            f"finished={request.is_finished}"
        )

        if not request.is_finished:
            states.append(state)

    # ==================================================
    # Repeated Heterogeneous Decode
    # ==================================================

    iteration = 0

    print()
    print("=== HETEROGENEOUS DECODE ===")

    while states:
        output = runner.decode_step(states)

        print(
            f"iteration={iteration:02d} "
            f"batch={list(output.request_ids)} "
            f"input={output.input_shape} "
            f"logical="
            f"{list(output.logical_cache_lengths_before)} "
            f"physical="
            f"{output.physical_cache_length}"
        )

        # Finished Request 不参加下一轮。
        #
        # 注意这里没有 admission 新 Request，
        # 因此还不是完整 Continuous Batching。
        states = [state for state in states if not state.request.is_finished]

        iteration += 1

    # ==================================================
    # Correctness
    # ==================================================

    print()
    print("=== CORRECTNESS ===")

    for request in requests:
        reference = references[request.request_id]

        actual = request.generated_token_ids

        identical = actual == reference

        print(f"{request.request_id}: {identical}")

        print(
            "  actual:   ",
            actual,
        )

        print(
            "  reference:",
            reference,
        )

        if not identical:
            raise AssertionError(
                f"Request {request.request_id} does not match HF generate()."
            )

    print()
    print("Repeated heterogeneous decode: PASS")


if __name__ == "__main__":
    main()
