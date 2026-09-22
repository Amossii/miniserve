import platform
import sys

import torch
import transformers


def main():
    print("=== MiniServe Environment Check ===")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Platform:     {platform.platform()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Torch CUDA:   {torch.version.cuda}")
    print(f"CUDA usable:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. "
            "Do not continue MiniServe setup yet."
        )

    device = torch.device("cuda")
    print(f"GPU:          {torch.cuda.get_device_name(device)}")

    # Small real GPU computation.
    x = torch.randn(1024, 1024, device=device)
    y = x @ x

    torch.cuda.synchronize()

    print("GPU test:     OK")
    print(f"Result shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
