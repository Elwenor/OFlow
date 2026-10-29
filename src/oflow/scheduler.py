"""One bounded scheduler shared by all interfaces; only trusted backends execute code."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .config import SchedulerConfig
from .contracts import (
    TERMINAL,
    TRANSITIONS,
    CostBackend,
    ErrorCategory,
    ExecutionCancelled,
    ExecutionContext,
    ExecutionPolicy,
    ExecutionProfile,
    ResourceLease,
    RoutingDecision,
    TaskError,
    TaskRequest,
    TaskResult,
    TaskState,
    TenantIdentity,
    WorkerCrashed,
)
from .registry import RequestFailure, RulesRouter, TaskRegistry
from .resources import ResourceManager
from .telemetry import Telemetry
from .workers import LocalWorker, WorkerSupervisor


class FairPriorityPolicy:
    """Tenant round-robin is enforced by the scheduler; aging bounds priority starvation."""

    def __init__(self, aging_seconds: float):
        self.aging_seconds = aging_seconds

    def rank(
        self,
        requests: list[tuple[TaskRequest, float]],
        profiles: dict[str, ExecutionProfile],
        now: float,
    ) -> list[int]:
        def score(index):
            request, admitted = requests[index]
            return (
                request.priority
                + profiles[request.profile].priority_boost
                + int((now - admitted) / self.aging_seconds),
                -admitted,
            )

        return sorted(range(len(requests)), key=score, reverse=True)


@dataclass
class Job:
    request: TaskRequest
    tenant: TenantIdentity
    decision: RoutingDecision | None
    future: asyncio.Future
    created: float
    expires: float
    context: ExecutionContext
    state: TaskState = TaskState.PENDING
    started: float | None = None
    finished: float | None = None
    worker: str | None = None
    physical_running: bool = False
    costs: dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return self.tenant.id, self.request.request_id

    def transition(self, state: TaskState) -> None:
        if state not in TRANSITIONS.get(self.state, set()):
            raise RuntimeError(f"invalid task transition {self.state} -> {state}")
        self.state = state


class TaskHandle:
    def __init__(self, scheduler: Scheduler, job: Job):
        self._scheduler, self._job = scheduler, job

    @property
    def request_id(self) -> str:
        return self._job.request.request_id

    @property
    def state(self) -> TaskState:
        return self._job.state

    async def result(self) -> TaskResult:
        try:
            return await asyncio.shield(self._job.future)
        except asyncio.CancelledError:
            self.cancel()
            raise

    def cancel(self) -> bool:
        return self._scheduler.cancel_job(self._job)


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        profiles: dict[str, ExecutionProfile],
        tasks: TaskRegistry,
        router: RulesRouter,
        supervisor: WorkerSupervisor,
        resources: ResourceManager,
        telemetry: Telemetry,
        policy: ExecutionPolicy | None = None,
        audit_callback=None,
    ):
        self.config, self.profiles, self.tasks = config, profiles, tasks
        self.router, self.supervisor, self.resources = router, supervisor, resources
        self.telemetry, self.audit_callback = telemetry, audit_callback
        self.policy = policy or FairPriorityPolicy(config.aging_seconds)
        self.queue: list[Job] = []
        self.records: OrderedDict[tuple[str, str], Job] = OrderedDict()
        self.running: set[asyncio.Task] = set()
        self.tenant_running: Counter = Counter()
        self.tenant_queued: Counter = Counter()
        self.inflight = 0
        self.last_tenant: str | None = None
        self.accepting = False
        self.stopping = False
        self.wake = asyncio.Event()
        self.dispatcher: asyncio.Task | None = None

    async def start(self) -> None:
        self.accepting = True
        self.dispatcher = asyncio.create_task(self._dispatch_loop(), name="oflow-dispatch")

    def _new_job(self, request: TaskRequest, tenant: TenantIdentity) -> Job:
        now = time.monotonic()
        timeout = self.config.max_request_seconds
        if request.deadline:
            timeout = min(timeout, (request.deadline - datetime.now(UTC)).total_seconds())
        return Job(
            request,
            tenant,
            None,
            asyncio.get_running_loop().create_future(),
            now,
            now + timeout,
            ExecutionContext(request.request_id, tenant, "cpu", deadline=request.deadline),
        )

    async def submit(self, request: TaskRequest, tenant: TenantIdentity) -> TaskHandle:
        self._prune()
        job = self._new_job(request, tenant)
        try:
            if not self.accepting:
                raise RequestFailure(
                    ErrorCategory.UNAVAILABLE, "Engine is not accepting work", True
                )
            if job.key in self.records:
                raise RequestFailure(
                    ErrorCategory.CONFLICT, "Request ID already exists for this tenant"
                )
            if request.profile not in self.profiles:
                raise RequestFailure(ErrorCategory.INVALID_REQUEST, "Unknown execution profile")
            if len(request.model_dump_json().encode()) > self.config.max_input_bytes:
                raise RequestFailure(
                    ErrorCategory.INVALID_REQUEST, "Request exceeds configured size"
                )
            job.request = self.tasks.validate_request(request)
            job.decision = self.router.route(job.request, tenant)
            for name in job.decision.candidates:
                backend = self.supervisor.workers[name].backend
                if backend.descriptor.batching == "cost":
                    try:
                        assert isinstance(backend, CostBackend)
                        cost = backend.estimate_cost(job.request)
                        if type(cost) is not int or cost < 1:
                            raise ValueError("invalid cost estimate")
                        job.costs[name] = cost
                    except Exception as exc:
                        raise RequestFailure(
                            ErrorCategory.BACKEND, "Backend cost estimation failed"
                        ) from exc
            if not any(
                self.resources.can_ever_fit(
                    self.supervisor.workers[name].backend.descriptor.requirements
                )
                for name in job.decision.candidates
            ):
                raise RequestFailure(
                    ErrorCategory.UNAVAILABLE, "Request cannot fit configured resource budgets"
                )
            if not any(self.supervisor.workers[name].healthy for name in job.decision.candidates):
                raise RequestFailure(
                    ErrorCategory.UNAVAILABLE, "Compatible workers are unavailable", True
                )
            if job.expires <= time.monotonic():
                job.transition(TaskState.TIMED_OUT)
                self._finish(
                    job,
                    error=TaskError(
                        category=ErrorCategory.DEADLINE, message="Deadline expired before admission"
                    ),
                )
                return TaskHandle(self, job)
            if (
                len(self.queue) >= self.config.queue_limit
                or self.tenant_queued[tenant.id] >= tenant.queue_limit
            ):
                raise RequestFailure(ErrorCategory.OVERLOADED, "Queue capacity exceeded", True)
            job.transition(TaskState.ADMITTED)
            self.records[job.key] = job
            self.queue.append(job)
            self.tenant_queued[tenant.id] += 1
            self.wake.set()
        except RequestFailure as exc:
            job.transition(TaskState.REJECTED)
            self._finish(
                job,
                error=TaskError(
                    category=exc.category, message=exc.message, retryable=exc.retryable
                ),
            )
        return TaskHandle(self, job)

    def _remove_queued(self, job: Job) -> None:
        self.queue.remove(job)
        self.tenant_queued[job.tenant.id] -= 1
        if self.tenant_queued[job.tenant.id] == 0:
            del self.tenant_queued[job.tenant.id]

    def _finish(self, job: Job, *, output=None, error: TaskError | None = None) -> None:
        if job.future.done():
            return
        now = time.monotonic()
        job.finished = now
        descriptor = self.supervisor.workers[job.worker].backend.descriptor if job.worker else None
        result = TaskResult(
            request_id=job.request.request_id,
            status=job.state,
            output=output,
            backend=job.worker,
            backend_version=descriptor.version if descriptor else None,
            queue_seconds=(job.started or now) - job.created,
            execution_seconds=now - job.started if job.started else 0,
            total_seconds=now - job.created,
            error=error,
            metadata={
                "correlation_id": job.request.correlation_id,
                "routing": job.decision.model_dump(mode="json") if job.decision else None,
                "worker": f"local:{job.worker}" if job.worker else None,
            },
        )
        metadata = self.telemetry.record(
            result,
            task=job.request.task,
            tenant=job.tenant.id,
            profile=job.request.profile,
            worker=job.worker,
        )
        if self.audit_callback:
            self.audit_callback(metadata)
        job.future.set_result(result)
        if not job.physical_running:
            self._forget_input(job)
        self._prune()

    @staticmethod
    def _forget_input(job: Job) -> None:
        job.request = job.request.model_copy(update={"input": {}, "options": {}})

    def _prune(self) -> None:
        now = time.monotonic()
        terminal_keys = [
            key
            for key, job in self.records.items()
            if job.state in TERMINAL and not job.physical_running
        ]
        excess = max(0, len(terminal_keys) - self.config.history_limit)
        for index, key in enumerate(terminal_keys):
            job = self.records[key]
            if index < excess or (
                job.finished is not None and now - job.finished > self.config.history_ttl_seconds
            ):
                del self.records[key]

    def cancel_job(self, job: Job, *, timeout: bool = False) -> bool:
        if job.state in TERMINAL:
            return False
        job.context.cancelled.set()
        if job.state == TaskState.ADMITTED:
            self._remove_queued(job)
        job.transition(TaskState.TIMED_OUT if timeout else TaskState.CANCELLED)
        self._finish(
            job,
            error=TaskError(
                category=ErrorCategory.DEADLINE if timeout else ErrorCategory.CANCELLED,
                message="Execution deadline exceeded" if timeout else "Execution cancelled",
            ),
        )
        self.wake.set()
        return True

    def status(self, request_id: str, tenant: TenantIdentity) -> TaskState | None:
        self._prune()
        job = self.records.get((tenant.id, request_id))
        return job.state if job else None

    def cancel(self, request_id: str, tenant: TenantIdentity) -> bool:
        job = self.records.get((tenant.id, request_id))
        return self.cancel_job(job) if job else False

    def _ranked_jobs(self, now: float) -> list[Job]:
        tenants = list(dict.fromkeys(j.tenant.id for j in self.queue))
        if self.last_tenant in tenants:
            pivot = tenants.index(self.last_tenant) + 1
            tenants = tenants[pivot:] + tenants[:pivot]
        ranked: list[Job] = []
        for tenant in tenants:
            jobs = [
                j
                for j in self.queue
                if j.tenant.id == tenant
                and self.tenant_running[tenant] < j.tenant.concurrency_limit
            ]
            try:
                order = self.policy.rank([(j.request, j.created) for j in jobs], self.profiles, now)
                if sorted(order) != list(range(len(jobs))):
                    raise ValueError("scheduling policy returned invalid ordering")
            except Exception:
                self.accepting = False
                for pending in list(self.queue):
                    self._remove_queued(pending)
                    pending.transition(TaskState.FAILED)
                    self._finish(
                        pending,
                        error=TaskError(
                            category=ErrorCategory.INTERNAL, message="Scheduling policy failed"
                        ),
                    )
                return []
            ranked.extend(jobs[i] for i in order)
        return ranked

    def _select_worker(self, job: Job) -> tuple[LocalWorker, ResourceLease] | None:
        assert job.decision is not None
        for name in job.decision.candidates:
            worker = self.supervisor.workers[name]
            descriptor = worker.backend.descriptor
            if not worker.healthy or worker.active >= descriptor.max_concurrency:
                continue
            lease = self.resources.acquire(descriptor.requirements)
            if lease:
                return worker, lease
        return None

    def _batch(self, first: Job, worker: LocalWorker, now: float) -> list[Job]:
        descriptor = worker.backend.descriptor
        profile = self.profiles[first.request.profile]
        limit = min(
            profile.max_batch_size,
            descriptor.max_batch_size,
            self.config.max_inflight - self.inflight,
            first.tenant.concurrency_limit - self.tenant_running[first.tenant.id],
        )
        if descriptor.batching == "none":
            limit = 1
        jobs, cost = [first], 0
        if descriptor.batching == "cost":
            cost = first.costs[worker.config.id]
        for job in self._ranked_jobs(now):
            if len(jobs) >= limit:
                break
            if job is first or job.tenant.id != first.tenant.id:
                continue
            if (
                job.request.task != first.request.task
                or job.request.profile != first.request.profile
            ):
                continue
            assert job.decision is not None
            if (
                job.request.options != first.request.options
                or worker.config.id not in job.decision.candidates
            ):
                continue
            item_cost = job.costs[worker.config.id] if descriptor.batching == "cost" else 1
            if descriptor.batching == "cost" and cost + item_cost > profile.cost_budget:
                continue
            jobs.append(job)
            cost += item_cost
        return jobs

    async def _dispatch_loop(self) -> None:
        while not self.stopping:
            self.wake.clear()
            now = time.monotonic()
            for job in list(self.records.values()):
                if job.state not in TERMINAL and job.expires <= now:
                    self.cancel_job(job, timeout=True)
            self._prune()
            while self.inflight < self.config.max_inflight:
                selected = None
                for job in self._ranked_jobs(now):
                    assert job.decision is not None
                    delay = self.profiles[job.request.profile].batch_delay_ms / 1000
                    if now - job.created < delay and job.expires - now > delay:
                        continue
                    selected = self._select_worker(job)
                    if selected:
                        break
                    if not any(
                        self.supervisor.workers[name].healthy for name in job.decision.candidates
                    ):
                        self._remove_queued(job)
                        job.transition(TaskState.FAILED)
                        self._finish(
                            job,
                            error=TaskError(
                                category=ErrorCategory.UNAVAILABLE,
                                message="Compatible workers are unavailable",
                            ),
                        )
                if selected is None:
                    break
                worker, lease = selected
                jobs = self._batch(job, worker, now)
                self.last_tenant = job.tenant.id
                worker.active += 1
                self.inflight += len(jobs)
                for item in jobs:
                    self._remove_queued(item)
                    item.transition(TaskState.RUNNING)
                    item.started, item.worker, item.physical_running = (
                        time.monotonic(),
                        worker.config.id,
                        True,
                    )
                    item.context.device = worker.config.device
                    assert item.decision is not None
                    item.decision = item.decision.model_copy(
                        update={"selected_backend": worker.config.id}
                    )
                    self.tenant_running[item.tenant.id] += 1
                task = asyncio.create_task(
                    self._run(worker, jobs, lease), name=f"oflow:{worker.config.id}"
                )
                self.running.add(task)
                task.add_done_callback(self.running.discard)
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=self.config.poll_seconds)
            except TimeoutError:
                pass

    async def _run(self, worker: LocalWorker, jobs: list[Job], lease: ResourceLease) -> None:
        crashed = False
        try:
            with self.telemetry.span(
                backend=worker.config.id,
                task=jobs[0].request.task,
                profile=jobs[0].request.profile,
                batch_size=len(jobs),
                request_id=jobs[0].request.request_id,
            ):
                outputs = await worker.execute([j.request for j in jobs], [j.context for j in jobs])
            if len(outputs) != len(jobs):
                raise ValueError("backend returned incorrect batch cardinality")
            for job, output in zip(jobs, outputs, strict=True):
                if job.state in TERMINAL:
                    continue
                if time.monotonic() >= job.expires:
                    self.cancel_job(job, timeout=True)
                    continue
                try:
                    validated = self.tasks.validate_output(job.request.task, output)
                    if (
                        len(json.dumps(validated, allow_nan=False).encode())
                        > self.config.max_output_bytes
                    ):
                        raise ValueError("backend output exceeds configured size")
                except ValueError:
                    job.transition(TaskState.FAILED)
                    self._finish(
                        job,
                        error=TaskError(
                            category=ErrorCategory.BACKEND,
                            message="Backend output failed schema validation",
                        ),
                    )
                else:
                    job.transition(TaskState.COMPLETED)
                    self._finish(job, output=validated)
        except asyncio.CancelledError:
            for job in jobs:
                if job.state not in TERMINAL:
                    self.cancel_job(job)
            raise
        except Exception as exc:
            crashed = isinstance(exc, (WorkerCrashed, MemoryError)) or (
                type(exc).__name__ == "OutOfMemoryError"
                and type(exc).__module__.startswith("torch")
            )
            if not crashed and not isinstance(exc, ExecutionCancelled):
                try:
                    crashed = not (await worker._call(worker.backend.health)).healthy
                except Exception:
                    crashed = True
            if crashed:
                worker.healthy = False
                worker.unavailable_reason = "runtime_failed"
            for job in jobs:
                if job.state in TERMINAL:
                    continue
                if isinstance(exc, ExecutionCancelled) and job.context.cancelled.is_set():
                    self.cancel_job(job)
                else:
                    job.transition(TaskState.FAILED)
                    self._finish(
                        job,
                        error=TaskError(
                            category=ErrorCategory.BACKEND, message="Backend execution failed"
                        ),
                    )
        finally:
            # Never release reservations while a cancelled backend is still executing.
            self.resources.release(lease)
            worker.active -= 1
            self.inflight -= len(jobs)
            for job in jobs:
                job.physical_running = False
                self._forget_input(job)
                self.tenant_running[job.tenant.id] -= 1
                if self.tenant_running[job.tenant.id] == 0:
                    del self.tenant_running[job.tenant.id]
            self._prune()
            if not worker.healthy and worker.active == 0:
                try:
                    await worker.restart()
                except Exception:
                    worker.healthy = False
            self.wake.set()

    async def close(self) -> None:
        self.accepting = False
        deadline = time.monotonic() + self.config.shutdown_grace_seconds
        while (self.queue or self.running) and time.monotonic() < deadline:
            await asyncio.sleep(min(0.01, max(0, deadline - time.monotonic())))
        for job in list(self.records.values()):
            if job.state not in TERMINAL:
                self.cancel_job(job)
        # Threads/CUDA cannot be killed safely: graceful close waits for physical completion.
        if self.running:
            await asyncio.gather(*list(self.running))
        self.stopping = True
        self.wake.set()
        if self.dispatcher:
            await self.dispatcher
