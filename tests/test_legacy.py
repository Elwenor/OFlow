import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from oflow import OFlowEngine, TaskRequest, TaskState
from oflow.benchmark import equivalent
from oflow.config import load_config
from oflow.contracts import ExecutionContext, TenantIdentity
from oflow.legacy.adapter import InferenceSettings
from oflow.legacy.pipeline import exact_batches, infer
from oflow.legacy.semantic import semantic_one_doc


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, max_length, **kwargs):
        ids = [[len(word) + 1 for word in text.split()][:max_length] or [1] for text in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


def tiny_model(torch):
    class Model(torch.nn.Module):
        def forward(self, ids, mask, return_embedding=True):
            sums = (ids * mask).sum(dim=1).float()
            return {
                "decision": torch.stack([sums, -sums], dim=1),
                "embedding": torch.stack([sums, mask.sum(dim=1).float()], dim=1),
            }

    return Model().eval()


def test_legacy_cpu_pipeline_is_bounded_and_preserves_association():
    torch = pytest.importorskip("torch")
    settings = InferenceSettings(
        max_length=32, token_budget=64, max_batch=4, tokenize_window=4, prefetch_depth=2
    )
    items = [(i, "one " * (i % 7 + 1)) for i in range(31)]
    context = ExecutionContext("test", TenantIdentity("test"), "cpu")
    keys, output, counts = infer(
        tiny_model(torch),
        TinyTokenizer(),
        items,
        device=torch.device("cpu"),
        want="embedding",
        settings=settings,
        context=context,
    )
    assert sorted(keys) == list(range(len(items)))
    assert output.device.type == "cpu" and not output.requires_grad
    for key, value, count in zip(keys, output.tolist(), counts, strict=True):
        expected_count = key % 7 + 1
        assert value == [expected_count * 4, expected_count]
        assert count == expected_count
    batches = list(
        exact_batches(
            TinyTokenizer(),
            items,
            max_length=32,
            token_budget=64,
            max_batch=4,
            window=4,
            workers=1,
            depth=2,
            pad_multiple=8,
            min_efficiency=0.5,
            efficiency_min_batch=1,
        )
    )
    assert all(batch[2]["input_ids"].numel() <= 64 for batch in batches)


def test_semantic_geometry_analytical_cases():
    torch = pytest.importorskip("torch")
    selector = SimpleNamespace(words=lambda text: text.split())
    empty = semantic_one_doc(selector, [], [])
    assert empty["paragraphs_embedded"] == 0 and not empty["coherence_observable"]
    paragraphs = ["same paragraph words " * 50 for _ in range(4)]
    embeddings = [
        (0, torch.tensor([1.0, 0.0])),
        (1, torch.tensor([1.0, 0.0])),
        (2, torch.tensor([0.0, 1.0])),
        (3, torch.tensor([0.0, 1.0])),
    ]
    result = semantic_one_doc(selector, paragraphs, embeddings)
    assert result["adjacent_cosine_mean"] == pytest.approx(2 / 3)
    assert result["adjacent_cosine_min"] == 0
    assert len(result["semantic_regions"]) == 2
    assert result["largest_coherent_region_ratio"] == 0.5
    assert result["best_recoverable_split"]["before_paragraph"] == 2
    assert result["recoverable_split_score"] == 1


def test_extracted_semantic_matches_preserved_original():
    torch = pytest.importorskip("torch")
    original_path = Path(__file__).resolve().parents[1] / "legacy/oflow_v5_beta.original.py"
    spec = importlib.util.spec_from_file_location("oflow_original_test", original_path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        generator = torch.Generator().manual_seed(42)
        selector = SimpleNamespace(words=lambda text: text.split())
        paragraphs = ["reference geometry " * (i + 50) for i in range(9)]
        vectors = [(i, torch.randn(8, generator=generator)) for i in range(9)]
        expected = module._semantic_fast_one_doc(selector, paragraphs, vectors)
        assert equivalent(expected, semantic_one_doc(selector, paragraphs, vectors), atol=1e-8)
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.cuda
def test_cuda_pipeline_equivalence_and_repeated_memory():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("physical CUDA device is not available")
    settings = InferenceSettings(max_length=32, token_budget=128, max_batch=8)
    items = [(i, "physical gpu test " * (i % 8 + 1)) for i in range(64)]
    context = ExecutionContext("cuda", TenantIdentity("test"), "cuda:0")
    cpu_keys, cpu_output, _ = infer(
        tiny_model(torch),
        TinyTokenizer(),
        items,
        device=torch.device("cpu"),
        want="embedding",
        settings=settings,
        context=context,
    )
    baseline = torch.cuda.memory_allocated()
    for _ in range(10):
        keys, outputs, _ = infer(
            tiny_model(torch).cuda(),
            TinyTokenizer(),
            items,
            device=torch.device("cuda:0"),
            want="embedding",
            settings=settings,
            context=context,
        )
        assert keys == cpu_keys
        assert torch.allclose(outputs, cpu_output)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= baseline + 1024 * 1024


@pytest.mark.legacy
def test_external_selector_reference_when_assets_are_available():
    config_path = os.environ.get("OFLOW_LEGACY_CONFIG")
    reference_path = os.environ.get("OFLOW_LEGACY_REFERENCE_JSON")
    if not config_path or not reference_path:
        pytest.skip(
            "set OFLOW_LEGACY_CONFIG and OFLOW_LEGACY_REFERENCE_JSON for real-model regression"
        )
    pytest.importorskip("torch")
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))

    async def scenario():
        async with OFlowEngine(load_config(config_path)) as engine:
            result = await engine.execute(
                TaskRequest(task="legacy.semantic", input={"texts": reference["texts"]})
            )
            assert result.status == TaskState.COMPLETED
            assert equivalent(reference["output"], result.output)

    asyncio.run(scenario())
