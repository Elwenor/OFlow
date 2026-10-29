import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fastapi.testclient import TestClient

from oflow import OFlowEngine, TaskRequest, TaskState
from oflow.api import create_app
from oflow.config import EngineConfig

KEY_A = "a" * 40
KEY_B = "b" * 40


@pytest.fixture
def api_config(monkeypatch):
    monkeypatch.setenv("OFLOW_TEST_A", KEY_A)
    monkeypatch.setenv("OFLOW_TEST_B", KEY_B)
    return EngineConfig.model_validate(
        {
            "backends": [
                {"id": "cpu", "kind": "cpu"},
                {"id": "mock", "kind": "mock", "options": {"delay_seconds": 0.04}},
            ],
            "tenants": [
                {"id": "alpha", "api_key_env": "OFLOW_TEST_A", "admin": True},
                {
                    "id": "beta",
                    "api_key_env": "OFLOW_TEST_B",
                    "tasks": ["text.score"],
                    "backends": ["cpu"],
                    "queue_limit": 2,
                    "concurrency_limit": 1,
                },
            ],
            "telemetry": {"structured_logs": False},
        }
    )


def headers(key=KEY_A):
    return {"Authorization": f"Bearer {key}"}


def payload(**changes):
    return {"task": "text.score", "input": {"text": "Ala ma kota"}, **changes}


def test_api_requires_secrets_and_authentication(api_config, monkeypatch):
    monkeypatch.delenv("OFLOW_TEST_B")
    with pytest.raises(ValueError):
        with TestClient(create_app(api_config)):
            pass
    with pytest.raises(ValueError):
        with TestClient(create_app()):
            pass


def test_api_health_auth_execute_and_inventory(api_config):
    with TestClient(create_app(api_config)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").json() == {"ready": True}
        assert client.post("/v1/tasks/execute", json=payload()).status_code == 401
        assert client.get("/v1/backends", headers=headers("wrong")).status_code == 401
        result = client.post(
            "/v1/tasks/execute", json=payload(request_id="api-test"), headers=headers()
        )
        assert result.status_code == 200
        assert result.json()["status"] == "COMPLETED"
        assert result.headers["X-Request-ID"] == "api-test"
        assert result.json()["output"]["method"] == "lexical_diversity"
        assert client.get("/v1/tasks/api-test", headers=headers()).json()["status"] == "COMPLETED"
        assert client.get("/v1/tasks/api-test", headers=headers(KEY_B)).status_code == 404
        assert client.delete("/v1/tasks/api-test", headers=headers(KEY_B)).status_code == 404
        assert client.get("/metrics", headers=headers()).status_code == 200
        assert "oflow_latency_seconds_bucket" in client.get("/metrics", headers=headers()).text
        assert client.get("/metrics", headers=headers(KEY_B)).status_code == 403
        assert client.get("/v1/admin/workers", headers=headers(KEY_B)).status_code == 403
        assert client.get("/v1/admin/workers", headers=headers()).json()["devices"]
        inventory = client.get("/v1/backends", headers=headers(KEY_B)).json()
        assert [b["id"] for b in inventory["backends"]] == ["cpu"]
        assert "requirements" not in str(inventory)
        assert client.get("/docs").status_code in (401, 404)


@pytest.mark.parametrize(
    "body",
    [
        payload(tenant="alpha"),
        payload(priority=100),
        {"task": "math.statistics", "input": {"values": [1]}},
        payload(preferred_backend="mock"),
        payload(options={"plugin": "evil"}),
    ],
)
def test_api_authorization_and_identity_spoofing(api_config, body):
    with TestClient(create_app(api_config)) as client:
        result = client.post("/v1/tasks/execute", json=body, headers=headers(KEY_B))
        assert result.status_code in (403, 422)
        assert "error" in result.json()


def test_api_validation_does_not_reflect_sensitive_input(api_config):
    with TestClient(create_app(api_config)) as client:
        result = client.post(
            "/v1/tasks/execute",
            json={"task": 42, "input": {"secret": "DO-NOT-LOG"}},
            headers=headers(),
        )
        assert result.status_code == 422
        assert "DO-NOT-LOG" not in result.text
        result = client.post("/v1/tasks/execute", content="{invalid DO-NOT-LOG", headers=headers())
        assert result.status_code == 422
        assert "DO-NOT-LOG" not in result.text


def test_body_limit_and_batch_limit_and_unsupported_stream(api_config):
    config = api_config.model_copy(
        update={
            "api": api_config.api.model_copy(update={"max_body_bytes": 512, "max_batch_tasks": 1})
        }
    )
    with TestClient(create_app(config)) as client:
        assert (
            client.post("/v1/tasks/execute", content=b"x" * 513, headers=headers()).status_code
            == 413
        )
        assert (
            client.post(
                "/v1/tasks/batch", json={"requests": [payload(), payload()]}, headers=headers()
            ).status_code
            == 413
        )
        assert (
            client.post("/v1/tasks/stream", json=payload(), headers=headers()).json()["error"][
                "category"
            ]
            == "unsupported_capability"
        )
        result = client.post("/v1/tasks/batch", json={"requests": [payload()]}, headers=headers())
        assert result.json()["results"][0]["status"] == "COMPLETED"


def test_concurrent_authenticated_tenants_and_overload(api_config):
    async def scenario():
        app = create_app(api_config)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                results = await asyncio.gather(
                    *[
                        client.post(
                            "/v1/tasks/execute",
                            json=payload(),
                            headers=headers(KEY_A if i % 2 else KEY_B),
                        )
                        for i in range(12)
                    ]
                )
                assert all(r.status_code in (200, 429) for r in results)
                assert any(r.status_code == 200 for r in results[::2])
                assert any(r.status_code == 200 for r in results[1::2])
                assert any(r.status_code == 429 for r in results)
                assert len(app.state.engine.scheduler.queue) == 0

    asyncio.run(scenario())


def test_fixed_http_backend_with_local_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            encoded = json.dumps({"score": 0.7, "words": 2, "method": "test_http"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = EngineConfig.model_validate(
        {
            "backends": [
                {
                    "id": "remote",
                    "kind": "http",
                    "options": {
                        "url": f"http://127.0.0.1:{server.server_port}/infer",
                        "allow_insecure_localhost": True,
                    },
                }
            ]
        }
    )

    async def scenario():
        async with OFlowEngine(config) as engine:
            result = await engine.execute(
                TaskRequest(task="text.score", input={"text": "remote inference"})
            )
            assert result.status == TaskState.COMPLETED
            assert result.output["method"] == "test_http"

    try:
        asyncio.run(scenario())
        assert len(received) == 1
        assert "tenant" not in received[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://example.com",
        "https://user:secret@host/x",
        "https://host/x?url=bad",
    ],
)
def test_http_backend_rejects_unsafe_operator_urls(url):
    config = EngineConfig.model_validate(
        {"backends": [{"id": "remote", "kind": "http", "options": {"url": url}}]}
    )
    with pytest.raises(ValueError):
        OFlowEngine(config)


def test_streamed_body_limit_without_content_length(api_config):
    config = api_config.model_copy(
        update={"api": api_config.api.model_copy(update={"max_body_bytes": 256})}
    )

    async def scenario():
        app = create_app(config)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:

                async def chunks():
                    yield b"x" * 200
                    yield b"x" * 200

                response = await client.post(
                    "/v1/tasks/execute", content=chunks(), headers=headers()
                )
                assert response.status_code == 413

    asyncio.run(scenario())


def test_disconnect_cancels_backend_and_releases_resources(api_config):
    async def scenario():
        app = create_app(api_config)
        async with app.router.lifespan_context(app):
            messages = asyncio.Queue()
            await messages.put(
                {
                    "type": "http.request",
                    "body": json.dumps(
                        {
                            "request_id": "disconnect-test",
                            "task": "test.echo",
                            "input": {"value": "x"},
                            "profile": "latency_first",
                        }
                    ).encode(),
                    "more_body": False,
                }
            )
            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/tasks/execute",
                "raw_path": b"/v1/tasks/execute",
                "query_string": b"",
                "headers": [
                    (b"authorization", f"Bearer {KEY_A}".encode()),
                    (b"content-type", b"application/json"),
                ],
                "client": ("127.0.0.1", 1),
                "server": ("127.0.0.1", 8000),
            }
            sent = []

            async def send(message):
                sent.append(message)

            task = asyncio.create_task(app(scope, messages.get, send))
            engine = app.state.engine
            while not engine.scheduler.inflight:
                await asyncio.sleep(0.001)
            await messages.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, 2)
            while engine.scheduler.inflight:
                await asyncio.sleep(0.001)
            tenant = api_config.tenants[0].identity()
            assert engine.scheduler.status("disconnect-test", tenant) == TaskState.CANCELLED
            assert engine.resources.active["cpu"] == 0
            assert not engine.scheduler.queue

    asyncio.run(asyncio.wait_for(scenario(), 5))
