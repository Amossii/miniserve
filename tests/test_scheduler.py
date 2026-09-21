import pytest

from miniserve.request import (
    ExecutionPhase,
    Request,
)
from miniserve.scheduler import Scheduler


# --------------------------------------------------
# 创建一个简单测试 Request。
#
# 所有 Request 初始都应该：
#
# WAITING + PREFILL
# --------------------------------------------------
def make_request(
    request_id: str,
) -> Request:

    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=10,
    )


# --------------------------------------------------
# 验证新 Request 进入 waiting queue，
# add_request() 本身不会 admission。
# --------------------------------------------------
def test_add_request_enters_waiting_queue():

    scheduler = Scheduler(max_num_seqs=2)

    request = make_request("A")

    scheduler.add_request(request)

    assert scheduler.waiting_requests == (request,)

    assert scheduler.running_requests == ()

    assert request.is_waiting


# --------------------------------------------------
# 验证 schedule() 会将 waiting Request
# admission 到 running。
# --------------------------------------------------
def test_schedule_admits_waiting_requests():

    scheduler = Scheduler(max_num_seqs=2)

    request_a = make_request("A")
    request_b = make_request("B")

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    output = scheduler.schedule()

    assert output.newly_admitted == (
        request_a,
        request_b,
    )

    assert scheduler.waiting_requests == ()

    assert scheduler.running_requests == (
        request_a,
        request_b,
    )

    assert request_a.is_running
    assert request_b.is_running


# --------------------------------------------------
# 验证 max_num_seqs 容量限制。
#
# max_num_seqs = 2
#
# A/B 进入 running，
# C 必须继续 WAITING。
# --------------------------------------------------
def test_max_num_seqs_capacity():

    scheduler = Scheduler(max_num_seqs=2)

    request_a = make_request("A")
    request_b = make_request("B")
    request_c = make_request("C")

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)
    scheduler.add_request(request_c)

    scheduler.schedule()

    assert scheduler.running_requests == (
        request_a,
        request_b,
    )

    assert scheduler.waiting_requests == (request_c,)

    assert scheduler.num_free_slots == 0

    assert request_c.is_waiting


# --------------------------------------------------
# 验证 FIFO admission。
#
# 如果只能 admission 两个：
#
# A -> B -> C
#
# 应该得到：
#
# running = A, B
# waiting = C
# --------------------------------------------------
def test_fifo_admission():

    scheduler = Scheduler(max_num_seqs=2)

    request_a = make_request("A")
    request_b = make_request("B")
    request_c = make_request("C")

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)
    scheduler.add_request(request_c)

    output = scheduler.schedule()

    assert output.newly_admitted == (
        request_a,
        request_b,
    )

    assert scheduler.waiting_requests == (request_c,)


# --------------------------------------------------
# 验证 finished Request 会先被回收，
# 然后 waiting Request 立刻复用空出的 slot。
#
# 初始：
#
# running = A B C
# waiting = D
#
# B finish 后：
#
# schedule()
#
# running = A C D
# --------------------------------------------------
def test_finished_slot_is_reused():

    scheduler = Scheduler(max_num_seqs=3)

    request_a = make_request("A")
    request_b = make_request("B")
    request_c = make_request("C")
    request_d = make_request("D")

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)
    scheduler.add_request(request_c)
    scheduler.add_request(request_d)

    first_output = scheduler.schedule()

    assert first_output.scheduled_requests == (
        request_a,
        request_b,
        request_c,
    )

    # 模拟 execution layer 完成 B。
    request_b.mark_finished(timestamp=1.0)

    second_output = scheduler.schedule()

    assert second_output.newly_finished == (request_b,)

    assert second_output.newly_admitted == (request_d,)

    assert second_output.scheduled_requests == (
        request_a,
        request_c,
        request_d,
    )

    assert scheduler.running_requests == (
        request_a,
        request_c,
        request_d,
    )

    assert scheduler.finished_requests == (request_b,)


# --------------------------------------------------
# 验证 Scheduler 能正确区分
# PREFILL 和 DECODE Request。
# --------------------------------------------------
def test_schedule_partitions_prefill_and_decode():

    scheduler = Scheduler(max_num_seqs=3)

    request_a = make_request("A")
    request_b = make_request("B")

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    first_output = scheduler.schedule()

    # 两个新 Request 都应该需要 PREFILL。
    assert first_output.prefill_requests == (
        request_a,
        request_b,
    )

    assert first_output.decode_requests == ()

    # 模拟 execution layer 完成 A 的 prefill。
    request_a.mark_prefill_completed()

    second_output = scheduler.schedule()

    assert second_output.prefill_requests == (request_b,)

    assert second_output.decode_requests == (request_a,)

    assert request_a.phase is ExecutionPhase.DECODE


# --------------------------------------------------
# 验证没有 capacity 时不会产生新的 admission。
# --------------------------------------------------
def test_no_admission_when_full():

    scheduler = Scheduler(max_num_seqs=1)

    request_a = make_request("A")
    request_b = make_request("B")

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    scheduler.schedule()

    output = scheduler.schedule()

    assert output.newly_admitted == ()

    assert scheduler.running_requests == (request_a,)

    assert scheduler.waiting_requests == (request_b,)


# --------------------------------------------------
# 验证所有 Request 完成后，
# has_unfinished_requests() 最终为 False。
# --------------------------------------------------
def test_has_unfinished_requests():

    scheduler = Scheduler(max_num_seqs=1)

    request = make_request("A")

    scheduler.add_request(request)

    assert scheduler.has_unfinished_requests()

    scheduler.schedule()

    assert scheduler.has_unfinished_requests()

    request.mark_finished(timestamp=1.0)

    scheduler.schedule()

    assert not scheduler.has_unfinished_requests()


# --------------------------------------------------
# 重复 request_id 应该立即失败。
#
# 否则 metrics / trace / result lookup
# 很容易发生歧义。
# --------------------------------------------------
def test_duplicate_request_id_rejected():

    scheduler = Scheduler(max_num_seqs=2)

    scheduler.add_request(make_request("A"))

    with pytest.raises(ValueError):
        scheduler.add_request(make_request("A"))


# --------------------------------------------------
# 非 WAITING Request 不能作为新请求加入。
# --------------------------------------------------
def test_non_waiting_request_rejected():

    scheduler = Scheduler(max_num_seqs=2)

    request = make_request("A")

    request.mark_running()

    with pytest.raises(ValueError):
        scheduler.add_request(request)


# --------------------------------------------------
# max_num_seqs 必须为正数。
# --------------------------------------------------
def test_invalid_max_num_seqs():

    with pytest.raises(ValueError):
        Scheduler(max_num_seqs=0)
