"""Strict versioned operator configuration. No runtime/model imports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .contracts import Contract, ExecutionProfile, TenantIdentity


def default_profiles() -> dict[str, ExecutionProfile]:
    return {
        "latency_first": ExecutionProfile(name="latency_first", priority_boost=20),
        "balanced": ExecutionProfile(name="balanced", max_batch_size=4, batch_delay_ms=2),
        "throughput_first": ExecutionProfile(
            name="throughput_first", max_batch_size=16, batch_delay_ms=5
        ),
    }


class SchedulerConfig(Contract):
    queue_limit: int = Field(default=128, ge=1, le=100000)
    max_inflight: int = Field(default=32, ge=1, le=4096)
    history_limit: int = Field(default=256, ge=0, le=100000)
    history_ttl_seconds: float = Field(default=300, gt=0)
    poll_seconds: float = Field(default=0.01, gt=0, le=1)
    aging_seconds: float = Field(default=0.1, gt=0)
    max_request_seconds: float = Field(default=60, gt=0, le=86400)
    shutdown_grace_seconds: float = Field(default=10, ge=0)
    max_input_bytes: int = Field(default=1048576, ge=256)
    max_output_bytes: int = Field(default=1048576, ge=256)


class DeviceConfig(Contract):
    id: str = Field(pattern=r"^(cpu|cuda:[0-9]+)$")
    memory_mb: int | None = Field(default=None, ge=1)
    max_concurrency: int = Field(default=1, ge=1, le=4096)


class BackendConfig(Contract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.:-]+$", max_length=128)
    kind: str = "cpu"
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda:[0-9]+)$")
    concurrency: int = Field(default=1, ge=1, le=64)
    model_memory_mb: int = Field(default=0, ge=0)
    request_memory_mb: int = Field(default=0, ge=0)
    max_restarts: int = Field(default=1, ge=0, le=5)
    options: dict = Field(default_factory=dict)


class RoutingConfig(Contract):
    rules: dict[str, list[str]] = Field(default_factory=dict)


class TenantConfig(Contract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.:-]+$", max_length=128)
    api_key_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    concurrency_limit: int = Field(default=4, ge=1, le=4096)
    queue_limit: int = Field(default=32, ge=1, le=100000)
    tasks: frozenset[str] = frozenset({"*"})
    backends: frozenset[str] = frozenset({"*"})
    max_priority: int = Field(default=0, ge=-100, le=100)
    admin: bool = False

    def identity(self) -> TenantIdentity:
        return TenantIdentity(**self.model_dump(exclude={"api_key_env"}))


class APIConfig(Contract):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_body_bytes: int = Field(default=1048576, ge=256, le=67108864)
    max_concurrent_requests: int = Field(default=64, ge=1, le=4096)
    max_batch_tasks: int = Field(default=64, ge=1, le=1024)
    request_timeout_seconds: float = Field(default=65, gt=0, le=86400)
    docs_enabled: bool = False


class TelemetryConfig(Contract):
    structured_logs: bool = True
    tracing: bool = False
    metadata_sqlite: str | None = None
    audit_retention: int = Field(default=10000, ge=1, le=1000000)


class PluginConfig(Contract):
    name: str
    version: str
    api_version: Literal["1"] = "1"
    entry_point: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    dependencies: tuple[str, ...] = ()
    capabilities: tuple[str, ...]


class EngineConfig(Contract):
    schema_version: Literal["1"] = "1"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    devices: tuple[DeviceConfig, ...] = (DeviceConfig(id="cpu", max_concurrency=4),)
    backends: tuple[BackendConfig, ...] = (BackendConfig(id="cpu", concurrency=2),)
    profiles: dict[str, ExecutionProfile] = Field(default_factory=default_profiles)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    tenants: tuple[TenantConfig, ...] = ()
    api: APIConfig = Field(default_factory=APIConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    plugins: tuple[PluginConfig, ...] = ()
    trusted_plugin_allowlist: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def coherent(self) -> EngineConfig:
        for values, field in [
            (self.backends, "backends"),
            (self.devices, "devices"),
            (self.tenants, "tenants"),
        ]:
            ids = [v.model_dump()["id"] for v in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate IDs in {field}")
        if not self.backends or not self.devices:
            raise ValueError("at least one backend and device are required")
        devices = {d.id for d in self.devices}
        backends = {b.id for b in self.backends}
        if any(b.device not in devices for b in self.backends):
            raise ValueError("backend references an unconfigured device")
        if any(name != p.name for name, p in self.profiles.items()):
            raise ValueError("profile keys must match profile names")
        if not {"latency_first", "balanced", "throughput_first"}.issubset(self.profiles):
            raise ValueError("the three standard profiles must be configured")
        for task, targets in self.routing.rules.items():
            if not targets or not set(targets).issubset(backends):
                raise ValueError(f"invalid routing targets for {task}")
        for tenant in self.tenants:
            if "*" not in tenant.backends and not tenant.backends.issubset(backends):
                raise ValueError("tenant references an unknown backend")
        if any(p.name not in self.trusted_plugin_allowlist for p in self.plugins):
            raise ValueError("plugin is not explicitly allowlisted")
        return self


def load_config(path: str | Path | None = None, **overrides) -> EngineConfig:
    """Defaults < YAML < explicit top-level overrides. Secrets are env references only."""
    data = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping")
    data.update(overrides)
    return EngineConfig.model_validate(data)
