"""Small genuine CPU workloads, a deterministic test runtime and opt-in remote inference."""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Annotated, cast
from urllib.parse import urlsplit

from pydantic import Field, JsonValue

from .config import BackendConfig
from .contracts import (
    BackendDescriptor,
    BackendHealth,
    Contract,
    ExecutionCancelled,
    ExecutionContext,
    ResourceRequirements,
    TaskRequest,
)
from .registry import TaskRegistry

if TYPE_CHECKING:
    import httpx


class TextInput(Contract):
    text: str = Field(max_length=200000)


class ScoreOutput(Contract):
    score: float = Field(ge=0, le=1)
    words: int = Field(ge=0)
    method: str = Field(max_length=128)


class NumbersInput(Contract):
    values: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1, max_length=10000
    )


class StatisticsOutput(Contract):
    count: int
    mean: float
    minimum: float
    maximum: float


class EchoInput(Contract):
    value: JsonValue


class EchoOutput(Contract):
    value: JsonValue


class EmbeddingOutput(Contract):
    embedding: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1, max_length=256
    )
    dimensions: int = Field(ge=1, le=256)
    model: str


def register_builtin_tasks(tasks: TaskRegistry) -> None:
    tasks.register("text.score", TextInput, ScoreOutput)
    tasks.register("math.statistics", NumbersInput, StatisticsOutput)
    tasks.register("test.echo", EchoInput, EchoOutput)
    tasks.register("text.embed", TextInput, EmbeddingOutput)


class CPUBackend:
    def __init__(self, config: BackendConfig, tasks: TaskRegistry):
        if config.options:
            raise ValueError("CPU backend has no configurable options")
        if config.device != "cpu":
            raise ValueError("CPU backend requires the cpu device")
        self.descriptor = BackendDescriptor(
            id=config.id,
            version="1.0",
            capabilities=tuple(tasks.capability(t) for t in ("text.score", "math.statistics")),
            requirements=ResourceRequirements(
                device=config.device, memory_mb=config.request_memory_mb
            ),
            max_concurrency=config.concurrency,
            max_batch_size=32,
            batching="cost",
        )

    def initialize(self) -> None:
        pass

    def health(self) -> BackendHealth:
        return BackendHealth(healthy=True)

    def shutdown(self) -> None:
        pass

    def execute(self, request: TaskRequest, context: ExecutionContext) -> JsonValue:
        context.check_cancelled()
        if request.task == "text.score":
            words = str(request.input["text"]).casefold().split()
            return {
                "score": len(set(words)) / max(1, len(words)),
                "words": len(words),
                "method": "lexical_diversity",
            }
        values = cast(list[float], request.input["values"])
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    def execute_batch(
        self, requests: list[TaskRequest], contexts: list[ExecutionContext]
    ) -> list[JsonValue]:
        # Cooperative cancellation is per item; do not fail an entire batch for one cancelled row.
        outputs = []
        for request, context in zip(requests, contexts, strict=True):
            try:
                outputs.append(self.execute(request, context))
            except ExecutionCancelled:
                outputs.append(None)
        return outputs

    def estimate_cost(self, request: TaskRequest) -> int:
        return max(
            1,
            len(str(request.input.get("text", ""))) // 4
            if request.task == "text.score"
            else len(cast(list, request.input["values"])),
        )


class MockOptions(Contract):
    delay_seconds: float = Field(default=0, ge=0, le=10)
    score: float = Field(default=0.5, ge=0, le=1)
    fail: bool = False


class MockBackend:
    def __init__(self, config: BackendConfig, tasks: TaskRegistry):
        self.options = MockOptions.model_validate(config.options)
        if config.device != "cpu":
            raise ValueError(
                "mock backend requires cpu; use injected test workers for GPU simulation"
            )
        self.descriptor = BackendDescriptor(
            id=config.id,
            version="1.0",
            capabilities=tuple(tasks.capability(t) for t in ("text.score", "test.echo")),
            requirements=ResourceRequirements(memory_mb=config.request_memory_mb),
            max_concurrency=config.concurrency,
            max_batch_size=32,
            batching="fixed",
        )

    def initialize(self) -> None:
        pass

    def health(self) -> BackendHealth:
        return BackendHealth(healthy=True)

    def shutdown(self) -> None:
        pass

    def execute(self, request: TaskRequest, context: ExecutionContext) -> JsonValue:
        end = time.monotonic() + self.options.delay_seconds
        while time.monotonic() < end:
            if context.cancelled.wait(min(0.005, max(0, end - time.monotonic()))):
                context.check_cancelled()
        context.check_cancelled()
        if self.options.fail:
            raise RuntimeError("synthetic backend failure")
        if request.task == "test.echo":
            return {"value": request.input["value"]}
        return {
            "score": self.options.score,
            "words": len(str(request.input["text"]).split()),
            "method": "deterministic_mock",
        }

    def execute_batch(
        self, requests: list[TaskRequest], contexts: list[ExecutionContext]
    ) -> list[JsonValue]:
        outputs = []
        for request, context in zip(requests, contexts, strict=True):
            try:
                outputs.append(self.execute(request, context))
            except ExecutionCancelled:
                outputs.append(None)
        return outputs


class HTTPOptions(Contract):
    url: str
    tasks: tuple[str, ...] = ("text.score",)
    timeout_seconds: float = Field(default=10, gt=0, le=300)
    max_response_bytes: int = Field(default=1048576, ge=256, le=67108864)
    api_key_env: str | None = None
    allow_insecure_localhost: bool = False

    def checked_url(self) -> str:
        value = urlsplit(self.url)
        if value.username or value.password or value.query or value.fragment:
            raise ValueError("backend URL must not embed credentials, queries or fragments")
        local = value.hostname in {"127.0.0.1", "localhost", "::1"}
        if not value.hostname or (
            value.scheme != "https"
            and not (value.scheme == "http" and local and self.allow_insecure_localhost)
        ):
            raise ValueError(
                "remote backend requires HTTPS; localhost HTTP requires explicit opt-in"
            )
        return self.url


class HTTPBackend:
    """Fixed operator endpoint. No user-controlled URLs, redirects or environment proxies."""

    def __init__(self, config: BackendConfig, tasks: TaskRegistry):
        self.options = HTTPOptions.model_validate(config.options)
        self.url = self.options.checked_url()
        if config.device != "cpu":
            raise ValueError("HTTP transport must use cpu placement")
        self.descriptor = BackendDescriptor(
            id=config.id,
            version="1.0",
            runtime="http",
            capabilities=tuple(tasks.capability(t) for t in self.options.tasks),
            max_concurrency=config.concurrency,
            requirements=ResourceRequirements(memory_mb=config.request_memory_mb),
        )
        self.client: httpx.Client | None = None

    def initialize(self) -> None:
        import os

        import httpx

        headers = {}
        if self.options.api_key_env:
            secret = os.environ[self.options.api_key_env]
            if len(secret) < 16:
                raise ValueError("remote API key must contain at least 16 characters")
            headers["Authorization"] = f"Bearer {secret}"
        self.client = httpx.Client(
            timeout=self.options.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            headers=headers,
            limits=httpx.Limits(max_connections=self.descriptor.max_concurrency),
        )

    def health(self) -> BackendHealth:
        return BackendHealth(healthy=self.client is not None)

    def shutdown(self) -> None:
        if self.client:
            self.client.close()
            self.client = None

    def execute(self, request: TaskRequest, context: ExecutionContext) -> JsonValue:
        import json

        context.check_cancelled()
        if self.client is None:
            raise RuntimeError("transport is not initialized")
        # Deliberately do not forward trusted tenant context, secrets or plugin configuration.
        with self.client.stream("POST", self.url, json=request.model_dump(mode="json")) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                context.check_cancelled()
                size += len(chunk)
                if size > self.options.max_response_bytes:
                    raise ValueError("remote response exceeds configured limit")
                chunks.append(chunk)
        return json.loads(b"".join(chunks))


def torch_embedding_factory(config: BackendConfig, tasks: TaskRegistry):
    from .torch_backend import TorchEmbeddingBackend

    return TorchEmbeddingBackend(config, tasks)


BUILTIN_FACTORIES = {
    "cpu": CPUBackend,
    "mock": MockBackend,
    "http": HTTPBackend,
    "torch_embed": torch_embedding_factory,
}
