"""Experimental legacy integration; original selector and model assets must be supplied."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import Field, JsonValue, model_validator

from oflow.batching import bounded_map
from oflow.config import BackendConfig
from oflow.contracts import (
    BackendDescriptor,
    BackendHealth,
    Contract,
    ExecutionContext,
    ResourceRequirements,
    TaskRequest,
)
from oflow.registry import TaskRegistry

from .pipeline import import_trusted_file, infer, safe_checkpoint_metadata
from .semantic import semantic_one_doc


class LegacyInput(Contract):
    texts: list[str] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def bounded_texts(self):
        if any(len(t) > 200000 for t in self.texts):
            raise ValueError("legacy text exceeds size limit")
        return self


class LegacyOutput(Contract):
    documents: list[dict[str, JsonValue]]


class InferenceSettings(Contract):
    max_length: int = Field(default=256, ge=8, le=8192)
    token_budget: int = Field(default=8192, ge=8, le=1048576)
    max_batch: int = Field(default=32, ge=1, le=512)
    tokenize_window: int = Field(default=256, ge=1, le=4096)
    tokenizer_workers: int = Field(default=1, ge=1, le=8)
    prefetch_depth: int = Field(default=2, ge=1, le=8)
    pad_multiple: int = Field(default=8, ge=1, le=64)
    min_padding_efficiency: float = Field(default=0.8, ge=0, le=1)
    efficiency_min_batch: int = Field(default=4, ge=1, le=512)
    pin_memory: bool = True
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"

    @model_validator(mode="after")
    def valid_budget(self):
        if self.token_budget < self.max_length or self.max_length % self.pad_multiple:
            raise ValueError(
                "budget must fit max_length; max_length must be divisible by pad_multiple"
            )
        return self


class LegacyOptions(Contract):
    selector_script: str
    pipeline_script: str | None = None
    checkpoint: str
    tokenizer: str
    model_family: Literal["oris", "mmbert"]
    trust_external_code: bool = False
    document: InferenceSettings = InferenceSettings(max_length=1024)
    paragraph: InferenceSettings = InferenceSettings()
    max_paragraphs_per_doc: int = Field(default=96, ge=1, le=512)
    semantic_workers: int = Field(default=1, ge=1, le=8)
    semantic_prefetch: int = Field(default=2, ge=1, le=8)


class LegacyBackend:
    def __init__(self, config: BackendConfig, tasks: TaskRegistry):
        self.options = LegacyOptions.model_validate(config.options)
        if not self.options.trust_external_code:
            raise ValueError("legacy selector/pipeline code requires explicit operator trust")
        if config.concurrency != 1:
            raise ValueError(
                "experimental legacy backend supports one concurrent call per model instance"
            )
        for path in [self.options.selector_script, self.options.checkpoint, self.options.tokenizer]:
            if not Path(path).exists():
                raise ValueError(
                    "legacy assets are missing; supply selector, checkpoint and tokenizer"
                )
        if self.options.model_family == "oris" and (
            not self.options.pipeline_script or not Path(self.options.pipeline_script).is_file()
        ):
            raise ValueError("ORIS requires the original pipeline script")
        self.descriptor = BackendDescriptor(
            id=config.id,
            version="1.0-experimental",
            runtime="torch",
            capabilities=(tasks.capability("legacy.semantic"),),
            requirements=ResourceRequirements(
                device=config.device, memory_mb=config.request_memory_mb
            ),
        )
        self.selector: Any = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.schema: Any = None
        self.device: Any = None
        self.module_names: list[str] = []

    def initialize(self) -> None:
        import torch

        safe_checkpoint_metadata(self.options.checkpoint)
        suffix = uuid4().hex
        self.module_names.append(f"oflow_selector_{suffix}")
        self.selector = import_trusted_file(self.options.selector_script, self.module_names[-1])
        if self.options.model_family == "oris":
            assert self.options.pipeline_script is not None
            self.module_names.append(f"oflow_pipeline_{suffix}")
            pipeline = import_trusted_file(self.options.pipeline_script, self.module_names[-1])
            self.selector.load_mmbert = pipeline.make_oris_filter_loader()
        self.device = torch.device(self.descriptor.requirements.device)
        self.model, self.tokenizer, self.schema, _ = self.selector.load_mmbert(
            Path(self.options.checkpoint), self.options.tokenizer, self.device
        )
        self.model.eval()

    def health(self) -> BackendHealth:
        return BackendHealth(healthy=self.model is not None)

    def shutdown(self) -> None:
        import sys

        self.selector = self.model = self.tokenizer = self.schema = self.device = None
        for name in self.module_names:
            sys.modules.pop(name, None)
        self.module_names.clear()

    def execute(self, request: TaskRequest, context: ExecutionContext):
        import torch

        if self.model is None:
            raise RuntimeError("legacy model is not loaded")
        texts = cast(list[str], request.input["texts"])
        rows = list(enumerate(texts))
        keys, logits, counts = infer(
            self.model,
            self.tokenizer,
            rows,
            device=self.device,
            want="decision",
            settings=self.options.document,
            context=context,
        )
        if sorted(keys) != list(range(len(texts))):
            raise ValueError("document inference lost input association")
        probabilities = torch.softmax(logits, dim=-1).tolist()
        documents: list[dict[str, Any]] = [{} for _ in texts]
        for key, probs, count in zip(keys, probabilities, counts, strict=True):
            order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            if len(probs) != len(self.schema.decisions):
                raise ValueError("model labels differ from schema")
            documents[key] = {
                "old_mmbert": {
                    "decision_probs": dict(zip(self.schema.decisions, probs, strict=True)),
                    "predicted": self.schema.decisions[order[0]],
                    "confidence": probs[order[0]],
                    "second": self.schema.decisions[order[1]] if len(order) > 1 else None,
                    "second_probability": probs[order[1]] if len(order) > 1 else None,
                    "margin": probs[order[0]] - probs[order[1]] if len(order) > 1 else 1.0,
                },
                "tokenization": {
                    "mmbert_document_tokens_visible": count,
                    "mmbert_document_max_length": self.options.document.max_length,
                    "mmbert_document_truncated": count >= self.options.document.max_length,
                },
            }

        def process(item):
            index, text = item
            paragraphs = self.selector.paragraphs(text)
            limit = self.options.max_paragraphs_per_doc
            if len(paragraphs) > limit:
                selected = sorted(
                    {round(i * (len(paragraphs) - 1) / max(limit - 1, 1)) for i in range(limit)}
                )
                paragraphs = [paragraphs[i] for i in selected]
            para_items = [
                (i, p) for i, p in enumerate(paragraphs) if len(self.selector.words(p)) >= 3
            ]
            para_keys, embeddings, _ = infer(
                self.model,
                self.tokenizer,
                para_items,
                device=self.device,
                want="embedding",
                settings=self.options.paragraph,
                context=context,
            )
            return index, paragraphs, list(zip(para_keys, embeddings, strict=True))

        # GPU execution remains serial. Bounded semantic work for previous documents overlaps it.
        def geometry(item):
            index, paragraphs, embeddings = item
            return index, semantic_one_doc(self.selector, paragraphs, embeddings)

        with ThreadPoolExecutor(
            max_workers=self.options.semantic_workers, thread_name_prefix="oflow-semantic"
        ) as pool:
            for index, semantic in bounded_map(
                pool, geometry, (process(row) for row in rows), self.options.semantic_prefetch
            ):
                context.check_cancelled()
                documents[index]["semantic_structure"] = semantic
        return {"documents": documents}
