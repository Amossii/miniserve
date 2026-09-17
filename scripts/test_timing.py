import torch

from miniserve.benchmark import (
    benchmark_cuda_event,
    benchmark_wall_clock,
)


def main():
    device = "cuda"

    a = torch.randn(
        4096,
        4096,
        device=device,
    )

    b = torch.randn(
        4096,
        4096,
        device=device,
    )

    def matmul():
        return a @ b

    _, wall = benchmark_wall_clock(
        matmul,
        warmup=5,
        repeats=20,
    )

    _, cuda = benchmark_cuda_event(
        matmul,
        warmup=5,
        repeats=20,
    )

    print("=== Wall Clock ===")
    print(f"mean: {wall.mean_ms:.3f} ms")
    print(f"p50:  {wall.p50_ms:.3f} ms")

    print()

    print("=== CUDA Event ===")
    print(f"mean: {cuda.mean_ms:.3f} ms")
    print(f"p50:  {cuda.p50_ms:.3f} ms")


if __name__ == "__main__":
    main()