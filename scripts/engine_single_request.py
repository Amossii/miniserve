import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from miniserve.engine import Engine
from miniserve.request import Request

MODEL_ID = "/home/henry/project/models/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32


# ---------------------------------------------------------
# 加载模型，构造一个 Request，
# 然后完全通过 Engine.step() 完成生成。
#
# 最后和 Hugging Face generate() 做 token-level correctness
# comparison。
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

    input_ids = inputs["input_ids"].to(device)

    attention_mask = inputs["attention_mask"].to(device)

    prompt_length = input_ids.shape[1]

    # -----------------------------------------------------
    # MiniServe Engine
    # -----------------------------------------------------

    request = Request(
        request_id="request-0",
        prompt_token_ids=(input_ids[0].detach().cpu().tolist()),
        max_new_tokens=MAX_NEW_TOKENS,
    )

    engine = Engine(
        model=model,
        device=device,
    )

    engine.add_request(request)

    print("=== MiniServe Engine ===")
    print(f"Prompt length: {request.prompt_length}")
    print()

    step_index = 0

    while engine.has_unfinished_requests():
        phase_before = request.phase

        cache_before = engine.cache_length

        token_id = engine.step()

        print(
            f"step={step_index:02d} "
            f"phase={phase_before.name:<7} "
            f"token={token_id:<8} "
            f"cache_before={cache_before:<4} "
            f"cache_after={engine.cache_length:<4} "
            f"sequence={request.sequence_length}"
        )

        step_index += 1

    miniserve_generated_ids = torch.tensor(
        [request.generated_token_ids],
        dtype=torch.long,
        device=device,
    )

    miniserve_text = tokenizer.decode(
        request.generated_token_ids,
        skip_special_tokens=True,
    )

    print()
    print("=== MiniServe Output ===")
    print(miniserve_text)

    # -----------------------------------------------------
    # Hugging Face reference
    # -----------------------------------------------------

    with torch.inference_mode():
        hf_output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.0,
        )

    hf_generated_ids = hf_output_ids[
        :,
        prompt_length:,
    ]

    hf_text = tokenizer.decode(
        hf_generated_ids[0],
        skip_special_tokens=True,
    )

    print()
    print("=== Hugging Face Output ===")
    print(hf_text)

    # -----------------------------------------------------
    # Token-level correctness
    # -----------------------------------------------------

    identical = torch.equal(
        miniserve_generated_ids,
        hf_generated_ids,
    )

    print()
    print("=== Correctness ===")
    print(f"Token IDs identical: {identical}")

    if not identical:
        print()
        print("MiniServe IDs:")
        print(miniserve_generated_ids)

        print()
        print("HF IDs:")
        print(hf_generated_ids)

        raise AssertionError(
            "MiniServe Engine output differs from Hugging Face generate()."
        )


if __name__ == "__main__":
    main()
