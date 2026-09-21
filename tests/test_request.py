import pytest

from miniserve.request import (
    ExecutionPhase,
    Request,
    RequestStatus,
)


# 验证新创建 Request 的默认 lifecycle / phase。
def test_request_initial_state():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10, 20, 30],
        max_new_tokens=4,
        arrival_time=100.0,
    )

    assert request.status is RequestStatus.WAITING

    assert request.phase is ExecutionPhase.PREFILL

    assert request.is_waiting
    assert not request.is_running
    assert not request.is_finished

    assert request.needs_prefill
    assert not request.needs_decode

    assert request.prompt_length == 3

    assert request.num_generated_tokens == 0

    assert request.sequence_length == 3


# 验证合法 lifecycle：
#
# WAITING -> RUNNING
def test_request_can_start_running():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10],
    )

    request.mark_running()

    assert request.status is RequestStatus.RUNNING

    assert request.is_running

    assert not request.is_waiting


# 验证 WAITING request 不允许直接生成 token。
def test_waiting_request_cannot_generate_token():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10],
    )

    with pytest.raises(RuntimeError):
        request.append_generated_token(
            20,
            timestamp=1.0,
        )


# 验证 prefill 完成后：
#
# RUNNING + PREFILL
# ->
# RUNNING + DECODE
def test_prefill_to_decode_transition():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10],
    )

    request.mark_running()

    assert request.needs_prefill

    request.mark_prefill_completed()

    assert request.is_running

    assert not request.needs_prefill

    assert request.needs_decode


# 验证 WAITING request 不能直接切换到 DECODE。
def test_waiting_request_cannot_complete_prefill():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10],
    )

    with pytest.raises(RuntimeError):
        request.mark_prefill_completed()


# 验证 token append 和 TTFT。
def test_append_generated_token():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10, 20],
        max_new_tokens=3,
        arrival_time=100.0,
    )

    request.mark_running()

    request.append_generated_token(
        30,
        timestamp=100.25,
    )

    assert request.generated_token_ids == [30]

    assert request.num_generated_tokens == 1

    assert request.sequence_length == 3

    assert request.first_token_time == pytest.approx(100.25)

    assert request.ttft_seconds == pytest.approx(0.25)


# 验证 second token 不会覆盖 first_token_time。
def test_first_token_time_only_set_once():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[10],
        max_new_tokens=3,
        arrival_time=10.0,
    )

    request.mark_running()

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


# 验证 max_new_tokens。
def test_max_new_tokens():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2],
        max_new_tokens=2,
    )

    request.mark_running()

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


# 验证合法 lifecycle：
#
# RUNNING -> FINISHED
def test_finish_request():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
        arrival_time=50.0,
    )

    request.mark_running()

    request.mark_finished(
        timestamp=51.5,
    )

    assert request.status is RequestStatus.FINISHED

    assert request.is_finished

    assert not request.is_running

    assert request.e2e_latency_seconds == pytest.approx(1.5)


# 验证 WAITING request 不能直接 FINISHED。
def test_waiting_request_cannot_finish():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
    )

    with pytest.raises(RuntimeError):
        request.mark_finished(
            timestamp=1.0,
        )


# 验证 FINISHED request 不能再次被运行。
def test_finished_request_cannot_run_again():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
    )

    request.mark_running()

    request.mark_finished(
        timestamp=1.0,
    )

    with pytest.raises(RuntimeError):
        request.mark_running()


# 验证 FINISHED request 不允许继续生成 token。
def test_finished_request_cannot_generate():

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1],
    )

    request.mark_running()

    request.mark_finished(
        timestamp=1.0,
    )

    with pytest.raises(RuntimeError):
        request.append_generated_token(
            2,
            timestamp=2.0,
        )
