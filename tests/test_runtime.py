from __future__ import annotations

import pytest

from miniserve.runtime import resolve_token_budget, validate_engine_limits


def test_token_budget_defaults_distinguish_tiny_and_real_models():
    """输入模型路径状态；输出对应默认预算；确保所有 CLI 使用相同约定。"""
    assert resolve_token_budget(None, None) == 6
    assert resolve_token_budget("local/model", None) == 256
    assert resolve_token_budget("local/model", 99) == 99


def test_engine_limits_cover_capacity_and_full_prefill():
    """输入合法和非法限制；合法无返回，非法明确报错以保护调度 invariant。"""
    validate_engine_limits([[1, 2], [3]], max_running=2, token_budget=2)

    with pytest.raises(ValueError, match="max_running"):
        validate_engine_limits([[1]], max_running=3, token_budget=2)
    with pytest.raises(ValueError, match="Longest prompt"):
        validate_engine_limits([[1, 2, 3]], max_running=1, token_budget=2)
    with pytest.raises(ValueError, match="non-empty"):
        validate_engine_limits([], max_running=1, token_budget=2)
