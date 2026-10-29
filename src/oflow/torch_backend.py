"""Optional working PyTorch workload; deterministic random byte embeddings, not a trained model."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .config import BackendConfig
from .contracts import (
    BackendDescriptor,
    BackendHealth,
    Contract,
    ExecutionContext,
    ResourceRequirements,
    TaskRequest,
)
from .registry import TaskRegistry


class TorchOptions(Contract):
    dimensions: int = Field(default=16, ge=1, le=256)
    max_bytes: int = Field(default=2048, ge=1, le=8192)
    seed: int = Field(default=42, ge=0, le=2147483647)


class TorchEmbeddingBackend:
    def __init__(self, config: BackendConfig, tasks: TaskRegistry):
        self.options = TorchOptions.model_validate(config.options)
        self.descriptor = BackendDescriptor(
            id=config.id,
            version="toy-byte-embedding-1",
            runtime="torch",
            capabilities=(tasks.capability("text.embed"),),
            max_concurrency=config.concurrency,
            max_batch_size=32,
            batching="cost",
            requirements=ResourceRequirements(
                device=config.device, memory_mb=config.request_memory_mb
            ),
        )
        self.weights: Any = None

    def initialize(self) -> None:
        import torch

        generator = torch.Generator(device="cpu").manual_seed(self.options.seed)
        self.weights = torch.randn(256, self.options.dimensions, generator=generator).to(
            self.descriptor.requirements.device
        )

    def health(self) -> BackendHealth:
        return BackendHealth(healthy=self.weights is not None)

    def shutdown(self) -> None:
        self.weights = None

    def estimate_cost(self, request: TaskRequest) -> int:
        return max(1, min(self.options.max_bytes, len(str(request.input["text"]).encode("utf-8"))))

    def execute(self, request: TaskRequest, context: ExecutionContext):
        return self.execute_batch([request], [context])[0]

    def execute_batch(self, requests: list[TaskRequest], contexts: list[ExecutionContext]):
        import torch

        if self.weights is None:
            raise RuntimeError("embedding weights are not initialized")
        rows = [
            list(str(r.input["text"]).encode("utf-8")[: self.options.max_bytes]) or [0]
            for r in requests
        ]
        width = max(map(len, rows))
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows], device=self.weights.device)
        mask = torch.tensor(
            [[1] * len(r) + [0] * (width - len(r)) for r in rows], device=self.weights.device
        )
        with torch.inference_mode():
            embeddings = (self.weights[ids] * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            output = embeddings.float().cpu().tolist()
        return [
            None
            if c.cancelled.is_set()
            else {
                "embedding": row,
                "dimensions": self.options.dimensions,
                "model": "toy-byte-embedding-1",
            }
            for row, c in zip(output, contexts, strict=True)
        ]
