import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "/home/henry/project/models/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32
from miniserve.benchmark import (
    benchmark_wall_clock,
    tokens_per_second,
)


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MiniServe baseline.")

    device = torch.device("cuda")

    # Prefer BF16 when the GPU supports it.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("=== MiniServe Hugging Face Baseline ===")
    print(f"Model:  {MODEL_ID}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Dtype:  {dtype}")

    # --------------------------------------------------
    # 1. Load tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # --------------------------------------------------
    # 2. Load model
    # --------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)

    model.eval()

    # --------------------------------------------------
    # 3. Model information
    # --------------------------------------------------

    num_parameters = sum(p.numel() for p in model.parameters())

    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    print()
    print("=== Model Info ===")
    print(f"Parameters:       {num_parameters / 1e6:.2f} M")
    print(f"Parameter memory: {bytes_to_gib(parameter_bytes):.3f} GiB")
    print(f"CUDA allocated:  {bytes_to_gib(torch.cuda.memory_allocated()):.3f} GiB")

    # --------------------------------------------------
    # 4. Construct one request
    # --------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": ("你是猪吗？"),
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

    prompt_tokens = inputs["input_ids"].shape[1]

    print()
    print("=== Input ===")
    print(f"Prompt tokens: {prompt_tokens}")

    # --------------------------------------------------
    # 5. Generate
    # --------------------------------------------------

    def generate():
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

    output_ids, stats = benchmark_wall_clock(
        generate,
        warmup=3,
        repeats=10,
    )
    # --------------------------------------------------
    # 6. Extract generated tokens
    # --------------------------------------------------

    generated_ids = output_ids[:, prompt_tokens:]
    print(generated_ids)

    generated_tokens = generated_ids.shape[1]

    text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    print()
    print("=== Benchmark ===")

    print(f"Mean latency: {stats.mean_ms:.2f} ms")

    print(f"P50 latency:  {stats.p50_ms:.2f} ms")

    print(f"P95 latency:  {stats.p95_ms:.2f} ms")

    print(f"P99 latency:  {stats.p99_ms:.2f} ms")
    tps = tokens_per_second(
        generated_tokens,
        stats.mean_ms,
    )

    print(f"Output tok/s: {tps:.2f}")


if __name__ == "__main__":
    main()
