"""用 PyTorch Profiler 观察简单的 CUDA 矩阵乘。"""

import torch


def main() -> None:
    """创建两个矩阵，预热后采集十次矩阵乘，并打印 operator 统计。"""
    size = 2048
    left = torch.randn(size, size, device="cuda")
    right = torch.randn(size, size, device="cuda")

    # 预热用于排除首次 CUDA 初始化和 cuBLAS 初始化的影响。
    for _ in range(5):
        torch.mm(left, right)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        for _ in range(10):
            result = torch.mm(left, right)
        torch.cuda.synchronize()

    print(
        profile.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total",
            row_limit=10,
        )
    )
    print(f"result[0, 0] = {result[0, 0].item():.6f}")


if __name__ == "__main__":
    main()
