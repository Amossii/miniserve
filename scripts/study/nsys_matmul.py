"""供 nsys 观察的简单 CUDA 矩阵乘 workload。"""

import torch


def main() -> None:
    """创建两个矩阵，预热后在 NVTX 区间内执行十次矩阵乘。"""
    size = 2048
    left = torch.randn(size, size, device="cuda")
    right = torch.randn(size, size, device="cuda")

    # 预热不放进 NVTX 区间，因此 nsys 可以只关注稳定执行阶段。
    for _ in range(5):
        torch.mm(left, right)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("measured_matmul")
    for _ in range(10):
        result = torch.mm(left, right)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print(f"result[0, 0] = {result[0, 0].item():.6f}")


if __name__ == "__main__":
    main()
