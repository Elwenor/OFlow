# HTTP API v1

Install `.[api,http]` and start `oflow serve --config configs/api.yaml`. The service defaults to loopback. Secrets are environment variables, at least 32 characters each; no default credential exists. Use Authorization: Bearer KEY. Tenant identity comes exclusively from that key. Unknown task fields, including tenant/admin/import-path fields, fail validation.

| Endpoint | Authorization | Behavior |
|---|---|---|
| GET /health/live | Public | Process responds |
| GET /health/ready | Public | 200 if admission is open and at least one worker is healthy; otherwise 503 |
| GET /v1/capabilities | Tenant | Authorized task schemas |
| GET /v1/backends | Tenant | Filtered IDs, versions, allowed tasks, health; no filesystem/device details |
| POST /v1/tasks/execute | Tenant | Wait for one normalized TaskResult |
| POST /v1/tasks/batch | Tenant | Bounded list of requests, ordered results through shared SDK service |
| POST /v1/tasks/stream | Tenant | Explicit unsupported_capability; streaming is not implemented |
| GET /v1/tasks/{request_id} | Owning tenant | Retained status; 404 if expired, unknown or owned by another tenant |
| DELETE /v1/tasks/{request_id} | Owning tenant | Signal cancellation; returns cancelled boolean |
| GET /v1/admin/workers | Admin | Worker/device identity, capacity and reservation inventory |
| GET /metrics | Admin | Prometheus text counters, gauges, latency/queue histograms |

Readiness is aggregate, not a guarantee every task/backend is available. No plugin, config-file or checkpoint loader endpoint is exposed. OpenAPI/docs are disabled by default and authenticated when enabled.

## Request

```json
{
  "api_version": "1",
  "request_id": "example-001",
  "task": "text.score",
  "schema_version": "1",
  "input": {"text": "Example text"},
  "options": {},
  "profile": "latency_first",
  "priority": 0,
  "correlation_id": "example-call"
}
```

request_id defaults to a generated UUID. deadline is an optional timezone-aware ISO 8601 timestamp; max_request_seconds still applies. preferred_backend is optional and strict: an unavailable/incompatible preferred backend is not silently replaced. Priorities are integers -100..100 and may not exceed the authenticated tenant's configured maximum. Task-specific input/output schemas come from /v1/capabilities. Built-ins accept no user execution options.

Batch body is `{"requests": [TASK_REQUEST, TASK_REQUEST]}`. The endpoint caps item count before scheduler submission and uses at most four submission consumers. Batch responses are 200 when accepted at HTTP level; inspect each result's status/error. A batch is not transactional. All statuses are returned in the same order as input requests, even if execution order differs.

## Results and errors

TaskResult includes request_id, terminal status, output, backend/version, queue/execution/total seconds, usage (empty if unavailable), safe routing/worker/correlation metadata and an optional TaskError. Execute responses include X-Request-ID matching the task. Boundary/validation errors use generated IDs. Responses use Cache-Control: no-store.

Stable categories include invalid_request (422), unauthorized (401), forbidden (403), overloaded (429), unavailable (503), deadline_exceeded (504), cancelled (409), backend_failure (502), unsupported_capability (400), request_conflict (409), internal_error (500). A body exceeding the streaming size limit is 413. Errors never return exception messages, raw inputs, validation value dumps or tracebacks.

The scheduler bounds queue and physical request slots. The HTTP boundary separately limits concurrent requests and streamed body bytes, including chunked uploads without Content-Length. Request timeouts cover body receipt and execution. Disconnect detection is best-effort polling and cancellation is cooperative. A timed-out/disconnected client does not authorize early reuse of physical resources.

Status is ephemeral and bounded by retention; this is not a durable asynchronous jobs API. Duplicate IDs conflict only while retained, scoped per tenant. If response delivery fails after a completed operation, clients must not assume retries are exactly-once. Do not put irreversible external side effects in a backend without its own idempotency policy.
