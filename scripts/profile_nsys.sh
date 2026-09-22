#!/usr/bin/env bash
set -euo pipefail

# 从项目根目录运行。其余 profile_torch.py 参数会原样透传。
# 示例：bash scripts/profile_nsys.sh --model /path/to/model --device cuda
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys was not found in PATH." >&2
  exit 1
fi

mkdir -p benchmarks/profiles/nsys
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --force-overwrite=true \
  --output=benchmarks/profiles/nsys/miniserve \
  .venv/bin/python scripts/profile_torch.py \
  --device cuda \
  --output-dir benchmarks/profiles/torch_cuda \
  "$@"
