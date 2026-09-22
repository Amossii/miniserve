"""将一份 MiniServe benchmark 原始 JSON 转换为可审阅 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from miniserve.benchmark_report import render_markdown


def parse_args():
    """输入命令行；返回源 JSON 与目标 Markdown 路径；不读写文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="benchmark_serving.py 生成的 JSON")
    parser.add_argument("--output", type=Path, required=True, help="生成的 Markdown 报告")
    return parser.parse_args()


def main() -> None:
    """输入命令行路径；无返回；校验 JSON、聚合原始样本并写入可追溯报告。"""
    args = parse_args()
    payload = json.loads(args.input.read_text())
    report = render_markdown(payload, str(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
