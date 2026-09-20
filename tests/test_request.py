import pytest

from miniserve.request import Request


# --------------------------------------------------
# 验证一个新 Request 的基本初始状态。
# --------------------------------------------------
def test_request_initial_state():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10, 20, 30],
        max_new_tokens=4,
        arrival_time=100.0,
    )

    assert request.request_id == "req-1"

    assert request.prompt_length == 3

    assert request.num_generated_tokens == 0

    assert request.sequence_length == 3

    assert request.generated_token_ids == []

    assert request.all_token_ids == [
        10,
        20,
        30,
    ]

    assert request.first_token_time is None

    assert request.finish_time is None

    assert request.ttft_seconds is None

    assert request.e2e_latency_seconds is None

    assert not request.reached_max_new_tokens


# --------------------------------------------------
# 验证 append token 后，
# generated tokens 和 sequence length 正确更新。
# --------------------------------------------------
def test_append_generated_token():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10, 20],
        max_new_tokens=3,
        arrival_time=100.0,
    )

    request.append_generated_token(
        30,
        timestamp=100.25,
    )

    assert request.generated_token_ids == [30]

    assert request.num_generated_tokens == 1

    assert request.sequence_length == 3

    assert request.all_token_ids == [
        10,
        20,
        30,
    ]

    # 第一个生成 token 的时间就是 first_token_time。
    assert request.first_token_time == pytest.approx(100.25)

    assert request.ttft_seconds == pytest.approx(0.25)


# --------------------------------------------------
# 验证只有第一个生成 token 会设置 first_token_time。
#
# 后续 token 不应该修改 TTFT。
# --------------------------------------------------
def test_first_token_time_only_set_once():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10],
        max_new_tokens=3,
        arrival_time=10.0,
    )

    request.append_generated_token(
        20,
        timestamp=10.2,
    )

    request.append_generated_token(
        30,
        timestamp=10.5,
    )

    assert request.first_token_time == pytest.approx(10.2)

    assert request.ttft_seconds == pytest.approx(0.2)


# --------------------------------------------------
# 验证 max_new_tokens 限制。
# --------------------------------------------------
def test_max_new_tokens():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2],
        max_new_tokens=2,
    )

    request.append_generated_token(
        3,
        timestamp=1.0,
    )

    request.append_generated_token(
        4,
        timestamp=2.0,
    )

    assert request.reached_max_new_tokens

    with pytest.raises(RuntimeError):
        request.append_generated_token(
            5,
            timestamp=3.0,
        )


# --------------------------------------------------
# 验证 finish time 和 E2E latency。
# --------------------------------------------------
def test_finish_time_and_e2e_latency():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
        max_new_tokens=10,
        arrival_time=50.0,
    )

    request.mark_finished(
        timestamp=51.5,
    )

    assert request.finish_time == pytest.approx(51.5)

    assert request.e2e_latency_seconds == pytest.approx(1.5)


# --------------------------------------------------
# 一个 Request 不应该被重复 finish。
# --------------------------------------------------
def test_cannot_finish_twice():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
    )

    request.mark_finished(
        timestamp=1.0,
    )

    with pytest.raises(RuntimeError):
        request.mark_finished(
            timestamp=2.0,
        )
