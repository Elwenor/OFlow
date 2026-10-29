from concurrent.futures import ThreadPoolExecutor

import pytest

from oflow.batching import CostBatching, bounded_map


def test_cost_packing_bounds_order_association_and_efficiency():
    items = list(enumerate([100, 5, 90, 10, 80, 1, 70, 2]))
    policy = CostBatching(
        cost=lambda x: x[1], budget=240, max_batch=4, lookahead=4, min_efficiency=0.8
    )
    batches = list(policy.batches(items))
    assert sorted(x for batch in batches for x in batch) == items
    assert all(max(x[1] for x in batch) * len(batch) <= 240 for batch in batches)
    assert all(len(batch) <= 4 for batch in batches)
    assert all(
        sum(x[1] for x in batch) / (max(x[1] for x in batch) * len(batch)) >= 0.8
        for batch in batches
    )


def test_cost_rejects_oversized_item_and_bad_configuration():
    with pytest.raises(ValueError):
        list(CostBatching(cost=lambda x: x, budget=3, max_batch=2).batches([4]))
    with pytest.raises(ValueError):
        CostBatching(cost=lambda x: x, budget=0, max_batch=2)


def test_prefetch_is_ordered_and_bounded():
    consumed = []

    def source():
        for i in range(50):
            consumed.append(i)
            yield i

    with ThreadPoolExecutor(max_workers=2) as pool:
        iterator = bounded_map(pool, lambda x: x * 2, source(), depth=3)
        assert next(iterator) == 0
        assert len(consumed) == 3
        assert list(iterator) == list(range(2, 100, 2))
