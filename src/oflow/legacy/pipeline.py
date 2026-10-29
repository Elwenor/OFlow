"""Bounded replacement for v3/v4 inference staging; model-specific math stays here."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

from oflow.batching import CostBatching, bounded_map


def import_trusted_file(path: str, name: str):
    source = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import trusted selector/pipeline")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_checkpoint_metadata(path: str) -> dict:
    import torch

    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) < (2, 10):
        raise RuntimeError("Use a maintained PyTorch >=2.10 build for checkpoint handling")
    checkpoint = torch.load(Path(path).resolve(strict=True), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a tensor/state dictionary")
    return {key: checkpoint[key] for key in ("model_name", "args") if key in checkpoint}


def exact_batches(
    tokenizer,
    items,
    *,
    max_length: int,
    token_budget: int,
    max_batch: int,
    window: int,
    workers: int,
    depth: int,
    pad_multiple: int,
    min_efficiency: float,
    efficiency_min_batch: int,
):
    """At most depth windows are tokenized/padded ahead; input keys survive reordering."""
    import torch

    def windows():
        iterator = iter(items)
        while chunk := list(islice(iterator, window)):
            yield chunk

    def encode(chunk):
        encoding = tokenizer(
            [text for _, text in chunk],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
            verbose=False,
        )
        ids = encoding["input_ids"]
        masks = encoding.get("attention_mask", [[1] * len(row) for row in ids])
        if len(ids) != len(chunk) or len(masks) != len(chunk):
            raise ValueError("tokenizer returned incorrect cardinality")
        rows: list[tuple[Any, list[int], list[int]]] = []
        for (key, _), row, mask in zip(chunk, ids, masks, strict=True):
            if not row or len(row) != len(mask) or len(row) > max_length:
                raise ValueError("invalid tokenizer output")
            rows.append((key, list(row), list(mask)))
        packer: CostBatching[tuple[Any, list[int], list[int]]] = CostBatching(
            cost=lambda x: len(x[1]),
            budget=token_budget,
            max_batch=max_batch,
            lookahead=window,
            pad_multiple=pad_multiple,
            min_efficiency=min_efficiency,
            efficiency_min_batch=efficiency_min_batch,
        )
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None) or 0
        batches = []
        for batch in packer.batches(rows):
            length = max(len(r[1]) for r in batch)
            length = ((length + pad_multiple - 1) // pad_multiple) * pad_multiple
            # max_length is required to be divisible by pad_multiple by adapter validation.
            batches.append(
                (
                    [r[0] for r in batch],
                    [sum(r[2]) for r in batch],
                    {
                        "input_ids": torch.tensor(
                            [r[1] + [pad_id] * (length - len(r[1])) for r in batch]
                        ),
                        "attention_mask": torch.tensor(
                            [r[2] + [0] * (length - len(r[2])) for r in batch]
                        ),
                    },
                )
            )
        return batches

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="oflow-legacy-tokenizer"
    ) as pool:
        for batches in bounded_map(pool, encode, windows(), depth):
            yield from batches


def infer(model, tokenizer, items, *, device, want: str, settings, context):
    """Two staging slots maximum. CPU owns outputs after synchronized D2H per batch.

    Host input is retained until its copy event completes. GPU input records its
    compute stream. Failure synchronizes both streams before releasing buffers.
    D2H is intentionally blocking; overlap is bounded CPU tokenization/H2D/compute.
    """
    import torch

    keys, counts, outputs = [], [], []
    batches = iter(
        exact_batches(
            tokenizer,
            items,
            max_length=settings.max_length,
            token_budget=settings.token_budget,
            max_batch=settings.max_batch,
            window=settings.tokenize_window,
            workers=settings.tokenizer_workers,
            depth=settings.prefetch_depth,
            pad_multiple=settings.pad_multiple,
            min_efficiency=settings.min_padding_efficiency,
            efficiency_min_batch=settings.efficiency_min_batch,
        )
    )
    cuda = device.type == "cuda"
    copy_stream = torch.cuda.Stream(device=device) if cuda else None
    compute_stream = torch.cuda.current_stream(device) if cuda else None

    def stage(batch):
        batch_keys, batch_counts, cpu_inputs = batch
        if cuda:
            cpu_inputs = {
                k: v.pin_memory() if settings.pin_memory else v for k, v in cpu_inputs.items()
            }
            ready = torch.cuda.Event()
            with torch.cuda.stream(copy_stream):
                gpu_inputs = {
                    k: v.to(device, non_blocking=settings.pin_memory) for k, v in cpu_inputs.items()
                }
                ready.record(copy_stream)
            return batch_keys, batch_counts, cpu_inputs, gpu_inputs, ready
        return batch_keys, batch_counts, cpu_inputs, cpu_inputs, None

    staged = None
    try:
        raw = next(batches, None)
        staged = stage(raw) if raw is not None else None
        while staged is not None:
            context.check_cancelled()
            batch_keys, batch_counts, cpu_inputs, inputs, ready = staged
            if cuda:
                assert compute_stream is not None
                compute_stream.wait_event(ready)
                for tensor in inputs.values():
                    tensor.record_stream(compute_stream)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if settings.precision == "bf16" else torch.float16,
                    enabled=cuda and settings.precision != "fp32",
                ),
            ):
                result = model(
                    inputs["input_ids"], inputs["attention_mask"], return_embedding=True
                )[want]
            # The next H2D overlaps the current compute; retain both source slots until synchronized.
            raw = next(batches, None)
            following = stage(raw) if raw is not None else None
            cpu_result = (
                result.detach().float().cpu()
            )  # blocking D2H establishes safe output ownership
            if cuda:
                ready.synchronize()
            keys.extend(batch_keys)
            counts.extend(batch_counts)
            outputs.append(cpu_result)
            del result, inputs, cpu_inputs
            staged = following
        return keys, torch.cat(outputs) if outputs else torch.empty((0,)), counts
    finally:
        if cuda:
            # Includes exception/cancellation paths with a prefetched copy still in flight.
            assert copy_stream is not None and compute_stream is not None
            copy_stream.synchronize()
            compute_stream.synchronize()
        batches.close()
