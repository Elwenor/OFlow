"""OFlow 1.0-beta: model-independent local workload execution."""

from .contracts import TaskRequest, TaskResult, TaskState, TenantIdentity
from .engine import OFlowEngine, execute_sync

__version__ = "1.0.0b1"
__all__ = [
    "OFlowEngine",
    "TaskRequest",
    "TaskResult",
    "TaskState",
    "TenantIdentity",
    "execute_sync",
]
