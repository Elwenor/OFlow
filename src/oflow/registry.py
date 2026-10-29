"""Task schemas, backend declarations and deterministic authorization-aware routing."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from dataclasses import dataclass

from pydantic import BaseModel

from .config import EngineConfig
from .contracts import (
    Backend,
    BatchBackend,
    CostBackend,
    ErrorCategory,
    RoutingDecision,
    TaskCapability,
    TaskRequest,
    TenantIdentity,
)


class RequestFailure(Exception):
    def __init__(self, category: ErrorCategory, message: str, retryable: bool = False):
        super().__init__(message)
        self.category, self.message, self.retryable = category, message, retryable


@dataclass(frozen=True)
class TaskDefinition:
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    allowed_options: tuple[str, ...] = ()


class TaskRegistry:
    def __init__(self):
        self.definitions: dict[str, TaskDefinition] = {}

    def register(
        self,
        task: str,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        allowed_options: tuple[str, ...] = (),
    ) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z0-9_.]+", task):
            raise ValueError("task names must be namespaced")
        definition = TaskDefinition(input_model, output_model, allowed_options)
        if task in self.definitions and self.definitions[task] != definition:
            raise ValueError(f"conflicting schema registration: {task}")
        self.definitions[task] = definition

    def capability(self, task: str) -> TaskCapability:
        d = self.definitions[task]
        return TaskCapability(
            task=task,
            input_schema=d.input_model.model_json_schema(),
            output_schema=d.output_model.model_json_schema(),
            allowed_options=d.allowed_options,
        )

    def validate_request(self, request: TaskRequest) -> TaskRequest:
        definition = self.definitions.get(request.task)
        if definition is None:
            raise RequestFailure(ErrorCategory.UNSUPPORTED, "Task is not registered")
        if set(request.options) - set(definition.allowed_options):
            raise RequestFailure(ErrorCategory.INVALID_REQUEST, "Unsupported execution option")
        try:
            parsed = definition.input_model.model_validate(request.input)
        except ValueError as exc:
            raise RequestFailure(
                ErrorCategory.INVALID_REQUEST, "Input does not match task schema"
            ) from exc
        return request.model_copy(update={"input": parsed.model_dump(mode="json")}, deep=True)

    def validate_output(self, task: str, output):
        return self.definitions[task].output_model.model_validate(output).model_dump(mode="json")


class BackendRegistry:
    def __init__(self, tasks: TaskRegistry):
        self.tasks = tasks
        self.backends: dict[str, Backend] = {}

    def register(self, backend: Backend) -> None:
        if not isinstance(backend, Backend):
            raise ValueError("backend does not implement the lifecycle contract")
        d = backend.descriptor
        if d.id in self.backends:
            raise ValueError(f"duplicate backend: {d.id}")
        if not d.capabilities:
            raise ValueError("backend must declare at least one task")
        if d.batching != "none" and not isinstance(backend, BatchBackend):
            raise ValueError("batching declared without execute_batch")
        if d.batching == "cost" and not isinstance(backend, CostBackend):
            raise ValueError("cost batching declared without estimate_cost")
        if d.streaming:
            raise ValueError("streaming transport is not supported in this beta")
        for capability in d.capabilities:
            if capability.task not in self.tasks.definitions:
                raise ValueError(f"unregistered task: {capability.task}")
            if capability != self.tasks.capability(capability.task):
                raise ValueError("backend schema differs from registered task schema")
        self.backends[d.id] = backend


class RulesRouter:
    def __init__(self, registry: BackendRegistry, rules: dict[str, list[str]]):
        self.registry, self.rules = registry, rules
        for task, targets in rules.items():
            for target in targets:
                if not any(
                    c.task == task for c in registry.backends[target].descriptor.capabilities
                ):
                    raise ValueError(f"routing target {target} cannot execute {task}")

    def route(self, request: TaskRequest, tenant: TenantIdentity) -> RoutingDecision:
        if not tenant.permits_task(request.task) or request.priority > tenant.max_priority:
            raise RequestFailure(ErrorCategory.FORBIDDEN, "Task or priority is not permitted")
        if request.preferred_backend:
            ordered = [request.preferred_backend]  # explicit preference is strict
            if not tenant.permits_backend(request.preferred_backend):
                raise RequestFailure(ErrorCategory.FORBIDDEN, "Backend is not permitted")
        else:
            ordered = self.rules.get(request.task, sorted(self.registry.backends))
        candidates, reasons = [], []
        for name in ordered:
            backend = self.registry.backends.get(name)
            if backend is None or not tenant.permits_backend(name):
                continue
            if any(c.task == request.task for c in backend.descriptor.capabilities):
                candidates.append(name)
                reasons.append(f"{name}: schema-compatible and tenant-authorized; ordered rule")
        if not candidates:
            raise RequestFailure(ErrorCategory.UNAVAILABLE, "No permitted compatible backend")
        return RoutingDecision(candidates=tuple(candidates), reasons=tuple(reasons))


def load_plugins(config: EngineConfig, tasks: TaskRegistry) -> dict:
    """Operator-only code loading. Plugins run with all hosting-process privileges."""
    factories: dict = {}
    for plugin in config.plugins:
        if plugin.name not in config.trusted_plugin_allowlist or plugin.api_version != "1":
            raise ValueError("plugin trust/API version mismatch")
        for distribution in plugin.dependencies:
            importlib.metadata.version(distribution)
        module, name = plugin.entry_point.split(":")
        loaded = importlib.import_module(module)
        if getattr(loaded, "OFLOW_PLUGIN_API_VERSION", None) != plugin.api_version:
            raise ValueError("installed plugin has incompatible API version")
        if getattr(loaded, "OFLOW_PLUGIN_VERSION", None) != plugin.version:
            raise ValueError("installed plugin version differs from operator configuration")
        register = getattr(loaded, name)
        supplied = register(tasks)
        if not isinstance(supplied, dict) or set(supplied) - set(plugin.capabilities):
            raise ValueError("plugin factory names differ from declared capabilities")
        if set(supplied) & set(factories):
            raise ValueError("duplicate plugin factory")
        factories.update(supplied)
    return factories
