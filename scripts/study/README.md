# PyTorch Profiler 和 nsys

两个文件都执行同样的工作：预热 5 次，然后执行 10 次 `2048 x 2048`
CUDA 矩阵乘。它们故意不共用函数，方便直接比较两种工具各自需要什么代码。

## PyTorch Profiler

```bash
.venv/bin/python scripts/study/profile_matmul.py
```

它从 PyTorch 视角统计 operator。重点观察输出中的 `aten::mm`：调用次数、
CUDA 总耗时和输入 shape。

它主要回答：**哪个 PyTorch 算子耗时？**

## Nsight Systems（nsys）

```bash
mkdir -p benchmarks/profiles/study/nsys
nsys profile \
  --trace=cuda,nvtx,cublas \
  --output=benchmarks/profiles/study/nsys/matmul \
  .venv/bin/python scripts/study/nsys_matmul.py
```

代码中的 `measured_matmul` 是一个 NVTX 标记。用 Nsight Systems GUI 打开生成的
`matmul.nsys-rep`，在这个区间内观察 CPU 发起的 CUDA API 调用和 GPU 上执行的
矩阵乘 kernel。

它主要回答：**CPU 和 GPU 在时间线上是怎样配合的？GPU 中间有没有空闲？**

因此两者不是互相替代：PyTorch Profiler 更接近 framework/operator，nsys 更接近
整个进程与 GPU 的系统时间线。
