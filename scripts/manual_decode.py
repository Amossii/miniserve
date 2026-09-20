import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_ID = "/home/henry/project/models/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32


def normalize_eos_token_ids(
    eos_token_id: int | list[int] | None,
) -> set[int]:
    if eos_token_id is None:
        return set()

    if isinstance(eos_token_id, int):
        return {eos_token_id}

    return set(eos_token_id)


@torch.inference_mode()
def manual_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:

    eos_token_ids = normalize_eos_token_ids(model.generation_config.eos_token_id)

    # We keep the entire sequence here.
    current_input_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone()

    for step in range(max_new_tokens):
        # -----------------------------------------
        # 1. Run the entire sequence through model
        # -----------------------------------------

        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            use_cache=False,
        )

        # logits:
        #
        # [batch, sequence_length, vocab_size]
        #
        # We only need the distribution for
        # the token AFTER the current sequence.

        next_token_logits = outputs.logits[:, -1, :]

        # -----------------------------------------
        # 2. Greedy decoding
        # -----------------------------------------

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        # print(
        #     f"step={step:02d} "
        #     f"seq_len={current_input_ids.shape[1]} "
        #     f"next_token={next_token.item()}"
        # )

        # -----------------------------------------
        # 3. Append new token
        # -----------------------------------------

        current_input_ids = torch.cat(
            [
                current_input_ids,
                next_token,
            ],
            dim=1,
        )

        # The newly generated token is a real token,
        # so its attention mask value is 1.

        new_mask = torch.ones(
            (
                current_attention_mask.shape[0],
                1,
            ),
            dtype=current_attention_mask.dtype,
            device=current_attention_mask.device,
        )

        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                new_mask,
            ],
            dim=1,
        )

        # -----------------------------------------
        # 4. Stop on EOS
        # -----------------------------------------

        if next_token.item() in eos_token_ids:
            break

    return current_input_ids


def main():

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
            "content": ("你好，你能做什么？"),
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

    print("=== Prompt ===")
    print(f"Prompt length: {prompt_length}")
    print()

    # =========================================
    # Manual generation
    # =========================================

    manual_output = manual_generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    manual_generated_ids = manual_output[:, prompt_length:]

    manual_text = tokenizer.decode(
        manual_generated_ids[0],
        skip_special_tokens=True,
    )

    print()
    print("=== Manual Output ===")
    print(manual_text)

    # =========================================
    # Hugging Face reference
    # =========================================

    with torch.inference_mode():
        hf_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
            repetition_penalty=1.0,
        )

    hf_generated_ids = hf_output[:, prompt_length:]

    hf_text = tokenizer.decode(
        hf_generated_ids[0],
        skip_special_tokens=True,
    )

    print()
    print("=== Hugging Face Output ===")
    print(hf_text)

    # =========================================
    # Correctness check
    # =========================================

    same = torch.equal(
        manual_output,
        hf_output,
    )

    print()
    print("=== Correctness ===")
    print(f"Token IDs identical: {same}")

    if not same:
        print("Manual IDs:")
        print(manual_generated_ids)

        print("HF IDs:")
        print(hf_generated_ids)

        raise AssertionError(
            "Manual generation does not match Hugging Face generate()."
        )


if __name__ == "__main__":
    main()
