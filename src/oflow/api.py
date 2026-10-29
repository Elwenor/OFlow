"""Optional FastAPI adapter. Scheduling and authorization remain application services."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import Field
from starlette.exceptions import HTTPException

from .config import EngineConfig, load_config
from .contracts import Contract, ErrorCategory, TaskError, TaskRequest, TenantIdentity
from .engine import OFlowEngine

HTTP_ERRORS = {
    ErrorCategory.INVALID_REQUEST: 422,
    ErrorCategory.UNAUTHORIZED: 401,
    ErrorCategory.FORBIDDEN: 403,
    ErrorCategory.OVERLOADED: 429,
    ErrorCategory.UNAVAILABLE: 503,
    ErrorCategory.DEADLINE: 504,
    ErrorCategory.CANCELLED: 409,
    ErrorCategory.BACKEND: 502,
    ErrorCategory.UNSUPPORTED: 400,
    ErrorCategory.CONFLICT: 409,
    ErrorCategory.INTERNAL: 500,
}


def error_response(
    category: ErrorCategory, message: str, status: int | None = None, request_id: str | None = None
) -> JSONResponse:
    request_id = request_id or str(uuid4())
    return JSONResponse(
        {
            "api_version": "1",
            "request_id": request_id,
            "error": TaskError(category=category, message=message).model_dump(mode="json"),
        },
        status_code=status or HTTP_ERRORS[category],
        headers={"X-Request-ID": request_id},
    )


class APIKeys:
    def __init__(self, config: EngineConfig):
        if not config.tenants:
            raise ValueError("HTTP API requires at least one tenant and an environment API key")
        self.entries: list[tuple[bytes, TenantIdentity]] = []
        for tenant in config.tenants:
            secret = os.environ.get(tenant.api_key_env, "")
            if len(secret) < 32:
                raise ValueError(f"Set {tenant.api_key_env} to a secret of at least 32 characters")
            digest = hashlib.sha256(secret.encode()).digest()
            if any(hmac.compare_digest(digest, old) for old, _ in self.entries):
                raise ValueError("tenant API keys must be distinct")
            self.entries.append((digest, tenant.identity()))

    def authenticate(self, header: str) -> TenantIdentity | None:
        if not header.startswith("Bearer ") or len(header) > 4096:
            return None
        digest = hashlib.sha256(header[7:].encode()).digest()
        matched = None
        for expected, tenant in self.entries:
            if hmac.compare_digest(digest, expected):
                matched = tenant
        return matched


class BoundaryMiddleware:
    """ASGI-level streaming body limit; does not trust Content-Length or buffer unlimited bytes."""

    def __init__(self, app, *, config: EngineConfig, state):
        self.app, self.config, self.state = app, config, state
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        public = scope["path"] in {"/health/live", "/health/ready"}
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        auth = getattr(self.state, "auth", None)
        tenant = (
            auth.authenticate(headers.get(b"authorization", b"").decode("latin1")) if auth else None
        )
        if not public and tenant is None:
            await error_response(ErrorCategory.UNAUTHORIZED, "Valid Bearer API key required")(
                scope, receive, send
            )
            return
        if self.active >= self.config.api.max_concurrent_requests:
            await error_response(ErrorCategory.OVERLOADED, "HTTP admission limit exceeded")(
                scope, receive, send
            )
            return
        scope.setdefault("state", {})["tenant"] = tenant
        self.active += 1
        response_started = False

        async def tracked_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                hs = list(message.get("headers", []))
                if not any(k.lower() == b"x-request-id" for k, _ in hs):
                    hs.append((b"x-request-id", str(uuid4()).encode()))
                hs.append((b"cache-control", b"no-store"))
                message = {**message, "headers": hs}
            await send(message)

        async def bounded_request():
            body, total = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > self.config.api.max_body_bytes:
                    await error_response(
                        ErrorCategory.INVALID_REQUEST, "Request body too large", 413
                    )(scope, receive, tracked_send)
                    return
                body.append(chunk)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": b"".join(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, tracked_send)

        try:
            await asyncio.wait_for(bounded_request(), self.config.api.request_timeout_seconds)
        except TimeoutError:
            if not response_started:
                await error_response(ErrorCategory.DEADLINE, "HTTP request timed out")(
                    scope, receive, send
                )
        except Exception:
            if not response_started:
                await error_response(ErrorCategory.INTERNAL, "Internal server error")(
                    scope, receive, send
                )
        finally:
            self.active -= 1


class BatchRequest(Contract):
    requests: list[TaskRequest] = Field(min_length=1, max_length=1024)


async def until_disconnect(request: Request, operation):
    task = asyncio.create_task(operation)
    try:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                raise HTTPException(499, "Client disconnected")
            await asyncio.wait({task}, timeout=0.02)
        return await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def create_app(config: EngineConfig | None = None, *, engine: OFlowEngine | None = None) -> FastAPI:
    config = config or (engine.config if engine else EngineConfig())
    shared = engine or OFlowEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.auth = APIKeys(config)
        async with shared:
            yield

    app = FastAPI(
        title="OFlow",
        version="1.0.0-beta.1",
        lifespan=lifespan,
        docs_url="/docs" if config.api.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if config.api.docs_enabled else None,
    )
    app.state.engine = shared
    app.add_middleware(BoundaryMiddleware, config=config, state=app.state)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI defaults include rejected values; never reflect prompts or secrets.
        return error_response(ErrorCategory.INVALID_REQUEST, "Request does not match API schema")

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        category = (
            ErrorCategory.FORBIDDEN if exc.status_code == 403 else ErrorCategory.INVALID_REQUEST
        )
        return error_response(
            category,
            "Operation is not permitted"
            if exc.status_code == 403
            else "Requested operation is unavailable",
            exc.status_code,
        )

    @app.get("/health/live")
    async def live():
        return {"live": True}

    @app.get("/health/ready")
    async def ready():
        value = (
            shared.started
            and shared.scheduler.accepting
            and any(w.healthy for w in shared.supervisor.workers.values())
        )
        return JSONResponse({"ready": value}, status_code=200 if value else 503)

    @app.get("/v1/capabilities")
    async def capabilities(request: Request):
        tenant = request.state.tenant
        permitted = {
            c.task
            for name, backend in shared.registry.backends.items()
            if tenant.permits_backend(name)
            for c in backend.descriptor.capabilities
        }
        return {
            "api_version": "1",
            "tasks": [
                shared.tasks.capability(t).model_dump(mode="json")
                for t in sorted(permitted)
                if tenant.permits_task(t)
            ],
        }

    @app.get("/v1/backends")
    async def backends(request: Request):
        tenant = request.state.tenant
        return {
            "api_version": "1",
            "backends": [
                {
                    "id": name,
                    "version": b.descriptor.version,
                    "tasks": [
                        c.task for c in b.descriptor.capabilities if tenant.permits_task(c.task)
                    ],
                    "healthy": shared.supervisor.workers[name].healthy,
                }
                for name, b in shared.registry.backends.items()
                if tenant.permits_backend(name)
                and any(tenant.permits_task(c.task) for c in b.descriptor.capabilities)
            ],
        }

    @app.post("/v1/tasks/execute")
    async def execute(body: TaskRequest, request: Request):
        result = await until_disconnect(
            request, shared.execute(body, identity=request.state.tenant)
        )
        return JSONResponse(
            result.model_dump(mode="json"),
            status_code=HTTP_ERRORS[result.error.category] if result.error else 200,
            headers={"X-Request-ID": result.request_id},
        )

    @app.post("/v1/tasks/batch")
    async def batch(body: BatchRequest, request: Request):
        if len(body.requests) > config.api.max_batch_tasks:
            return error_response(ErrorCategory.INVALID_REQUEST, "Batch item limit exceeded", 413)
        results = await until_disconnect(
            request,
            shared.execute_many(
                body.requests,
                identity=request.state.tenant,
                concurrency=min(4, config.scheduler.max_inflight),
            ),
        )
        return {"api_version": "1", "results": [r.model_dump(mode="json") for r in results]}

    @app.post("/v1/tasks/stream")
    async def stream(body: TaskRequest, request: Request):
        if not request.state.tenant.permits_task(body.task):
            return error_response(ErrorCategory.FORBIDDEN, "Task is not permitted")
        return error_response(
            ErrorCategory.UNSUPPORTED, "Streaming is not implemented by this beta"
        )

    @app.get("/v1/tasks/{request_id}")
    async def status(request_id: str, request: Request):
        state = shared.scheduler.status(request_id, request.state.tenant)
        if state is None:
            raise HTTPException(404)
        return {"request_id": request_id, "status": state.value}

    @app.delete("/v1/tasks/{request_id}")
    async def cancel(request_id: str, request: Request):
        if shared.scheduler.status(request_id, request.state.tenant) is None:
            raise HTTPException(404)
        return {
            "request_id": request_id,
            "cancelled": shared.scheduler.cancel(request_id, request.state.tenant),
        }

    def require_admin(request: Request):
        if not request.state.tenant.admin:
            raise HTTPException(403)

    @app.get("/v1/admin/workers")
    async def workers(request: Request):
        require_admin(request)
        return {
            "workers": [
                w.snapshot().model_dump(mode="json") for w in shared.supervisor.workers.values()
            ],
            "devices": shared.resources.snapshot(),
        }

    @app.get("/metrics")
    async def metrics(request: Request):
        require_admin(request)
        text = shared.telemetry.prometheus(shared.scheduler, shared.supervisor, shared.resources)
        text += f"oflow_audit_dropped_total {shared.audit_dropped}\n"
        return PlainTextResponse(text, media_type="text/plain; version=0.0.4")

    return app


def app_from_env() -> FastAPI:
    """uvicorn oflow.api:app_from_env --factory; OFLOW_CONFIG is operator-controlled."""
    return create_app(load_config(os.environ.get("OFLOW_CONFIG")))
