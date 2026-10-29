"""Reusable bounded cost/padding-aware packing, independent of NLP and model runtimes."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from itertools import islice
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class BatchingStrategy(Protocol[T]):
    def batches(self, items: Iterable[T]) -> Iterator[list[T]]: ...


def bounded_map(
    executor: Executor, function: Callable[[T], R], items: Iterable[T], depth: int
) -> Iterator[R]:
    """Ordered prefetch with a strict upper bound on submitted futures, including results."""
    if depth < 1:
        raise ValueError("depth must be positive")
    iterator = iter(items)
    pending: deque[Future[R]] = deque()
    try:
        for item in islice(iterator, depth):
            pending.append(executor.submit(function, item))
        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()


@dataclass(frozen=True)
class CostBatching(Generic[T]):
    cost: Callable[[T], int]
    budget: int
    max_batch: int
    lookahead: int = 256
    pad_multiple: int = 1
    min_efficiency: float = 0
    efficiency_min_batch: int = 1

    def __post_init__(self):
        if (
            min(
                self.budget,
                self.max_batch,
                self.lookahead,
                self.pad_multiple,
                self.efficiency_min_batch,
            )
            < 1
            or not 0 <= self.min_efficiency <= 1
        ):
            raise ValueError("invalid batching configuration")

    def batches(self, items: Iterable[T]) -> Iterator[list[T]]:
        iterator = iter(items)
        while window := list(islice(iterator, self.lookahead)):
            costs = [(item, self.cost(item)) for item in window]
            if any(c < 1 for _, c in costs):
                raise ValueError("cost estimates must be positive")
            costs.sort(key=lambda row: row[1], reverse=True)
            chosen: list[T] = []
            total, padded = 0, 0
            for item, cost in costs:
                if cost > self.budget:
                    raise ValueError("a single item exceeds the cost budget")
                length = ((cost + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
                if length > self.budget:
                    raise ValueError("padded item exceeds the cost budget")
                next_padded = max(padded, length)
                efficiency = (total + cost) / ((len(chosen) + 1) * next_padded)
                if chosen and (
                    len(chosen) >= self.max_batch
                    or (len(chosen) + 1) * next_padded > self.budget
                    or (
                        len(chosen) >= self.efficiency_min_batch
                        and efficiency < self.min_efficiency
                    )
                ):
                    yield chosen
                    chosen, total, padded = [], 0, 0
                chosen.append(item)
                total += cost
                padded = max(padded, length)
            if chosen:
                yield chosen
