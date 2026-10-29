# Configuration

`oflow validate --config configs/cpu.yaml` validates YAML/Pydantic schemas without loading plugins, GPU models or checkpoints. Runtime registration additionally checks backend-specific options, capabilities and routing compatibility.

Precedence: built-in defaults < YAML < explicit top-level keyword overrides in load_config/from_config. Overrides replace whole top-level sections. `oflow serve --host/--port` overrides the API bind address after validation. Secrets are read from named environment variables at API/HTTP-backend startup. OFLOW_CONFIG chooses the config for the Uvicorn factory.

All models reject unknown fields. schema_version and public task/API schema versions are strings "1". Relative SQLite and operator-defined backend paths resolve against the process working directory; start from the project directory or configure absolute paths.

| Section | Main settings |
|---|---|
| scheduler | queue_limit, max_inflight, history_limit, history_ttl_seconds, max_request_seconds, poll_seconds, aging_seconds, shutdown_grace_seconds, max_input_bytes, max_output_bytes |
| devices | id, optional memory_mb cap, max_concurrency |
| backends | stable id, kind, device, concurrency, model_memory_mb, request_memory_mb, max_restarts, backend-specific options |
| profiles | name, max_batch_size, batch_delay_ms, cost_budget, priority_boost |
| routing.rules | task identifier → ordered backend IDs |
| tenants | id, api_key_env, queue_limit, concurrency_limit, tasks, backends, max_priority, admin |
| api | host, port, max_body_bytes, max_batch_tasks, max_concurrent_requests, request_timeout_seconds, docs_enabled |
| telemetry | structured_logs, optional tracing, metadata_sqlite, audit_retention |
| plugins | operator metadata: name, version, api_version, entry_point, dependencies, capabilities (factory names) |

Task schemas are registered by built-ins or trusted plugins, rather than arbitrary schema/import definitions supplied by remote users. No user-facing model-loader paths exist.

## Profiles

All three profiles share one scheduler. Defaults:

| Profile | Max batch | Delay | Priority boost | Tradeoff |
|---|---:|---:|---:|---|
| latency_first | 1 | 0 ms | 20 | Faster dispatch, fewer batching opportunities |
| balanced | 4 | 2 ms | 0 | Limited batching for mixed workloads |
| throughput_first | 16 | 5 ms | 0 | Larger batches, additional queue delay |

cost_budget defaults to 16384 and only applies when the backend declares cost batching. A single-item request still runs if its cost exceeds the profile target; the profile target is not a hard input/memory cap. Backends enforce actual task size and memory budgets. Actual batch sizes are further capped by worker declarations, remaining global/tenant request slots and compatible queued work.

Override profiles with all three names, for example:

```yaml
profiles:
  latency_first: {name: latency_first, max_batch_size: 2, batch_delay_ms: 0.5, priority_boost: 20}
  balanced: {name: balanced, max_batch_size: 4, batch_delay_ms: 2}
  throughput_first: {name: throughput_first, max_batch_size: 32, batch_delay_ms: 8, cost_budget: 32768}
```

No batch size or padding threshold is universally optimal; measure on the target workload.

Optional tracing uses OpenTelemetry API spans when telemetry.tracing is true and the telemetry extra is installed. Configure the tracer provider/exporter in the hosting application; OFlow does not install a collector or export payloads. SDK users configure Python logging handlers; the CLI server enables structured INFO logs.

## Limits and device budgets

The default global queue holds 128 requests; 32 requests may be physically in flight. Default terminal history retains up to 256 results for 300 seconds. Active/queued records are separate from that terminal limit. History can hold sensitive outputs in memory until eviction; set history_limit: 0 to disable retained terminal results. Handles still own their own results until released by the caller.

CPU visibility and slots work without torch. RAM discovery uses optional psutil; absent discovery and absent a configured memory cap, RAM memory accounting has no physical upper bound. Concurrency and queues are still bounded. GPU workers require a positive explicit model-memory budget. request_memory_mb is peak additional memory for a whole backend invocation, including its batch. Set conservative budgets and cap batches before loading large models.

Use [configs/gpu.yaml](configs/gpu.yaml) for one GPU or [configs/multi_gpu.yaml](configs/multi_gpu.yaml) for two independently placed optional PyTorch demonstration workers. A missing GPU leaves that worker unavailable while compatible CPU tasks continue. An unavailable text.embed request fails; it does not silently become a CPU request. GPU examples exit with an error when the requested device cannot execute. No weights are automatically duplicated except the backend entries explicitly configured by the operator.

## Runtime updates

Trusted same-process callers can use engine.update_routing({"text.score": ["mock", "cpu"]}). Changes validate atomically and affect new admissions; existing queued requests retain their routing decision. Backend/device/profile/tenant/key/plugin changes require recreating the engine. No public administrative mutation endpoint or general hot reload exists in beta.
