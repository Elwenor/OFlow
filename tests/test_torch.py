import asyncio

import pytest

from oflow import OFlowEngine, TaskRequest, TaskState
from oflow.config import EngineConfig


def test_optional_torch_backend_real_cpu_execution_and_rng_isolation():
    torch = pytest.importorskip("torch")
    initial = torch.random.get_rng_state().clone()
    config = EngineConfig.model_validate(
        {"backends": [{"id": "embeddings", "kind": "torch_embed"}]}
    )

    async def scenario():
        async with OFlowEngine(config) as engine:
            results = await engine.execute_many(
                [
                    TaskRequest(task="text.embed", input={"text": text})
                    for text in ["abc", "abc", "def"]
                ]
            )
            assert all(r.status == TaskState.COMPLETED for r in results)
            assert results[0].output == results[1].output
            assert results[0].output != results[2].output
            assert len(results[0].output["embedding"]) == 16

    asyncio.run(scenario())
    assert torch.equal(initial, torch.random.get_rng_state())
