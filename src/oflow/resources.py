"""Single-host inventory and conservative reservations; no torch import on CPU startup."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from uuid import uuid4

from .config import DeviceConfig
from .contracts import ResourceLease, ResourceRequirements


@dataclass(frozen=True)
class Device:
    id: str
    identity: str
    name: str
    memory_mb: int | None
    max_concurrency: int
    healthy: bool = True
    cuda_capability: tuple[int, int] | None = None
    runtime: str = "python"


def discover_devices(config: tuple[DeviceConfig, ...]) -> list[Device]:
    devices = []
    for entry in config:
        if entry.id == "cpu":
            memory = None
            try:
                import psutil

                memory = int(psutil.virtual_memory().available // 1048576)
            except ImportError:
                pass
            devices.append(
                Device(
                    "cpu",
                    "host-cpu",
                    os.environ.get("PROCESSOR_IDENTIFIER", "CPU"),
                    min(memory, entry.memory_mb)
                    if memory and entry.memory_mb
                    else entry.memory_mb or memory,
                    min(entry.max_concurrency, os.cpu_count() or 1),
                )
            )
            continue
        try:
            import torch  # explicit CUDA configuration is the opt-in to runtime probing

            index = int(entry.id.split(":")[1])
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise RuntimeError("CUDA device is unavailable")
            props = torch.cuda.get_device_properties(index)
            free, _ = torch.cuda.mem_get_info(index)
            memory = int(min(props.total_memory, free) // 1048576)
            identity = str(
                getattr(props, "uuid", None)
                or f"visible:{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}:{index}"
            )
            devices.append(
                Device(
                    entry.id,
                    identity,
                    props.name,
                    min(memory, entry.memory_mb or memory),
                    entry.max_concurrency,
                    cuda_capability=(props.major, props.minor),
                    runtime=f"torch:{torch.__version__}",
                )
            )
        except (ImportError, RuntimeError, AssertionError):
            devices.append(
                Device(
                    entry.id,
                    "unavailable",
                    "unavailable",
                    entry.memory_mb,
                    entry.max_concurrency,
                    healthy=False,
                    runtime="unavailable",
                )
            )
    return devices


class ResourceManager:
    """Event-loop-owned accounting. Estimates are budgets, not physical memory isolation."""

    def __init__(self, devices: list[Device]):
        self.devices = {d.id: d for d in devices}
        self.memory = dict.fromkeys(self.devices, 0)
        self.active = dict.fromkeys(self.devices, 0)
        self.cpu_active = 0
        self.cpu_capacity = (
            self.devices["cpu"].max_concurrency if "cpu" in self.devices else os.cpu_count() or 1
        )
        self.leases: dict[str, tuple[ResourceLease, bool]] = {}

    def can_ever_fit(self, requirements: ResourceRequirements) -> bool:
        device = self.devices.get(requirements.device)
        if device is None or not device.healthy or requirements.cpu_slots > self.cpu_capacity:
            return False
        persistent_mb = sum(
            lease.requirements.memory_mb
            for lease, persistent in self.leases.values()
            if persistent and lease.requirements.device == device.id
        )
        return (
            device.memory_mb is None or persistent_mb + requirements.memory_mb <= device.memory_mb
        )

    def acquire(
        self, requirements: ResourceRequirements, *, persistent: bool = False
    ) -> ResourceLease | None:
        d = self.devices.get(requirements.device)
        if d is None or not d.healthy:
            return None
        if d.memory_mb is not None and self.memory[d.id] + requirements.memory_mb > d.memory_mb:
            return None
        if not persistent and (
            self.active[d.id] >= d.max_concurrency
            or self.cpu_active + requirements.cpu_slots > self.cpu_capacity
        ):
            return None
        lease = ResourceLease(str(uuid4()), requirements)
        self.memory[d.id] += requirements.memory_mb
        if not persistent:
            self.active[d.id] += 1
            self.cpu_active += requirements.cpu_slots
        self.leases[lease.id] = (lease, persistent)
        return lease

    def release(self, lease: ResourceLease) -> None:
        saved = self.leases.pop(lease.id, None)
        if saved is None:
            return
        requirements = saved[0].requirements
        self.memory[requirements.device] -= requirements.memory_mb
        if not saved[1]:
            self.active[requirements.device] -= 1
            self.cpu_active -= requirements.cpu_slots

    def snapshot(self) -> list[dict]:
        return [
            {**asdict(d), "reserved_memory_mb": self.memory[d.id], "active": self.active[d.id]}
            for d in self.devices.values()
        ]
