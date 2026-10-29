"""Version 1 public contracts. No web framework or model runtime dependencies."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

API_VERSION = "1"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TaskState(StrEnum):
    PENDING = "PENDING"
    ADMITTED = "ADMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    REJECTED = "REJECTED"


TERMINAL = frozenset(
    {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.TIMED_OUT,
        TaskState.REJECTED,
    }
)
TRANSITIONS = {
    TaskState.PENDING: {TaskState.ADMITTED, TaskState.REJECTED, TaskState.TIMED_OUT},
    TaskState.ADMITTED: {
        TaskState.RUNNING,
        TaskState.CANCELLED,
        TaskState.TIMED_OUT,
        TaskState.FAILED,
    },
    TaskState.RUNNING: {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.TIMED_OUT,
    },
}


class ErrorCategory(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    OVERLOADED = "overloaded"
    UNAVAILABLE = "unavailable"
    DEADLINE = "deadline_exceeded"
    CANCELLED = "cancelled"
    BACKEND = "backend_failure"
    UNSUPPORTED = "unsupported_capability"
    CONFLICT = "request_conflict"
    INTERNAL = "internal_error"


class TaskError(Contract):
    category: ErrorCategory
    message: str
    retryable: bool = False


class TaskRequest(Contract):
    api_version: Literal["1"] = "1"
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_.:-]+$",
    )
    task: str = Field(min_length=3, max_length=128, pattern=r"^[a-z][a-z0-9_]*\.[a-z0-9_.]+$")
    schema_version: Literal["1"] = "1"
    input: dict[str, JsonValue]
    options: dict[str, JsonValue] = Field(default_factory=dict)
    profile: str = Field(default="balanced", min_length=1, max_length=64)
    deadline: datetime | None = None
    priority: int = Field(default=0, ge=-100, le=100, strict=True)
    preferred_backend: str | None = Field(default=None, min_length=1, max_length=128)
    correlation_id: str | None = Field(default=None, max_length=128, pattern=r"^[a-zA-Z0-9_.:-]+$")

    @field_validator("deadline")
    @classmethod
    def aware_deadline(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("deadline must include a timezone")
        return value


class TaskResult(Contract):
    api_version: Literal["1"] = "1"
    request_id: str
    status: TaskState
    output: JsonValue = None
    backend: str | None = None
    backend_version: str | None = None
    queue_seconds: float = 0
    execution_seconds: float = 0
    total_seconds: float = 0
    usage: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    error: TaskError | None = None


class TaskCapability(Contract):
    task: str
    schema_version: Literal["1"] = "1"
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    allowed_options: tuple[str, ...] = ()


class ResourceRequirements(Contract):
    device: str = "cpu"
    cpu_slots: int = Field(default=1, ge=0)
    memory_mb: int = Field(default=0, ge=0)


class BackendDescriptor(Contract):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_.:-]+$")
    version: str
    runtime: str = "thread"
    capabilities: tuple[TaskCapability, ...]
    requirements: ResourceRequirements = Field(default_factory=ResourceRequirements)
    max_concurrency: int = Field(default=1, ge=1, le=64)
    max_batch_size: int = Field(default=1, ge=1, le=256)
    batching: Literal["none", "fixed", "cost"] = "none"
    streaming: bool = False


class BackendHealth(Contract):
    healthy: bool
    reason: str = "ready"


class ExecutionProfile(Contract):
    name: str
    max_batch_size: int = Field(default=1, ge=1, le=256)
    batch_delay_ms: float = Field(default=0, ge=0, le=100)
    cost_budget: int = Field(default=16384, ge=1)
    priority_boost: int = Field(default=0, ge=-100, le=100)


class RoutingDecision(Contract):
    candidates: tuple[str, ...]
    reasons: tuple[str, ...]
    selected_backend: str | None = None


class WorkerDescriptor(Contract):
    id: str
    backend: str
    device: str
    healthy: bool
    active: int
    capacity: int
    restarts: int = 0
    reason: str = "ready"


@dataclass(frozen=True)
class TenantIdentity:
    """Trusted application context; never constructed from public task input."""

    id: str
    concurrency_limit: int = 4
    queue_limit: int = 32
    tasks: frozenset[str] = frozenset({"*"})
    backends: frozenset[str] = frozenset({"*"})
    max_priority: int = 0
    admin: bool = False

    def permits_task(self, task: str) -> bool:
        return "*" in self.tasks or task in self.tasks

    def permits_backend(self, backend: str) -> bool:
        return "*" in self.backends or backend in self.backends


@dataclass
class ExecutionContext:
    request_id: str
    tenant: TenantIdentity
    device: str
    cancelled: threading.Event = field(default_factory=threading.Event)
    deadline: datetime | None = None

    def check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise ExecutionCancelled()


@dataclass(frozen=True)
class ResourceLease:
    id: str
    requirements: ResourceRequirements


class ExecutionCancelled(Exception):
    pass


class WorkerCrashed(Exception):
    """A trusted backend signals that its persistent runtime must be recreated."""


@runtime_checkable
class Backend(Protocol):
    descriptor: BackendDescriptor

    def initialize(self) -> None: ...
    def execute(self, request: TaskRequest, context: ExecutionContext) -> JsonValue: ...
    def health(self) -> BackendHealth: ...
    def shutdown(self) -> None: ...


@runtime_checkable
class BatchBackend(Protocol):
    def execute_batch(
        self, requests: list[TaskRequest], contexts: list[ExecutionContext]
    ) -> list[JsonValue]: ...


@runtime_checkable
class CostBackend(Protocol):
    def estimate_cost(self, request: TaskRequest) -> int: ...


class ExecutionPolicy(Protocol):
    def rank(
        self,
        requests: list[tuple[TaskRequest, float]],
        profiles: dict[str, ExecutionProfile],
        now: float,
    ) -> list[int]: ...
