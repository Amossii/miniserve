from __future__ import annotations

from collections.abc import Iterable

from miniserve.engine import Engine
from miniserve.request import Request


class SequentialRunner:
    """
    多请求 Sequential Serving baseline。

    核心策略非常简单：

    Request A 完整执行
        ↓
    Request B 完整执行
        ↓
    Request C 完整执行

    同一时间 Engine 中最多只有一个 unfinished Request。

    这个实现不是性能优化版本，而是后续 Static Batching
    和 Continuous Batching 的 correctness / performance baseline。
    """

    # --------------------------------------------------
    # 保存底层单请求 Engine。
    #
    # SequentialRunner 不直接执行 model.forward()，
    # 仍然复用已经验证过的 Engine。
    # --------------------------------------------------
    def __init__(
        self,
        engine: Engine,
    ) -> None:

        self.engine = engine

    # --------------------------------------------------
    # 完整执行一个 Request。
    #
    # 输入：
    #     一个仍处于 WAITING 状态的 Request。
    #
    # 内部：
    #     加入 Engine，然后不断调用 engine.step()。
    #
    # 输出：
    #     同一个 Request 对象，但其状态已经变成 FINISHED，
    #     generated_token_ids 等字段也已经填充。
    # --------------------------------------------------
    def run_request(
        self,
        request: Request,
    ) -> Request:

        self.engine.add_request(request)

        while self.engine.has_unfinished_requests():
            self.engine.step()

        return request

    # --------------------------------------------------
    # Sequential 执行一组 Requests。
    #
    # 注意这里故意没有：
    # - batching
    # - concurrency
    # - scheduler
    #
    # 每个 request 都完整结束后，
    # 才会处理下一个。
    # --------------------------------------------------
    def run_requests(
        self,
        requests: Iterable[Request],
    ) -> list[Request]:

        finished_requests: list[Request] = []

        for request in requests:
            finished_request = self.run_request(request)

            finished_requests.append(finished_request)

        return finished_requests
