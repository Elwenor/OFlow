# Backends and trusted plugins

| Kind | Tasks | Dependency | Status |
|---|---|---|---|
| cpu | text.score (lexical diversity), math.statistics | Base installation | Implemented and tested |
| mock | test.echo, text.score (configured constant) | Base installation | Implemented and tested; synthetic only |
| http | Operator-selected registered tasks | .[http] | Implemented; local-server integration tested |
| torch_embed | text.embed | .[torch] | Working optional byte-embedding demonstration; random untrained weights |
| legacy plugin | legacy.semantic (ORIS/mmBERT) | Torch + original selector/pipeline/tokenizer/checkpoint | Experimental; real-model acceptance pending |

Backends declare stable IDs and versions, task schemas, runtime, resource requirements, concurrency, batching and streaming support. A backend's task declaration must exactly match the task registry. Lifecycle is initialize → execute/execute_batch → shutdown; health reports availability. The scheduler does not load model weights per request.

Optional capabilities are separate Protocols: BatchBackend.execute_batch returns one result per request in input order; CostBackend.estimate_cost returns a cheap positive workload estimate. Backends must be thread-safe at their declared concurrency. Model-specific semantics, tokenization and geometry never belong in Core.

## Add a task/backend

See [examples/custom_plugin.py](examples/custom_plugin.py) and [examples/custom.yaml](examples/custom.yaml) for a complete runnable uppercase plugin:

```bash
python -m examples.custom_plugin
```

A trusted plugin exports OFLOW_PLUGIN_API_VERSION and OFLOW_PLUGIN_VERSION; both must match operator config. Its entry point receives a TaskRegistry, registers typed input/output models and returns {factory_name: factory}. Factories accept (BackendConfig, TaskRegistry) and return a backend implementing the lifecycle. This adds tasks without touching the scheduler.

Install plugin packages through normal operator tooling. Name every allowed plugin in trusted_plugin_allowlist and configure its exact module:function entry point, version, API version, dependency distribution names and supplied factory names. Declared dependencies must already be installed. OFlow does not download plugins. An API user cannot choose an import path or register code. Version metadata is checked, but an allowlist/version is not a signature or supply-chain verification mechanism.

Plugins run with the privileges of the hosting Python process. No in-process sandbox or isolation between plugins exists. A bad plugin can block its thread, corrupt memory or crash the process. Explicit process isolation is future work.

## Fixed-URL HTTP transport

Configure options.url, optional api_key_env, timeout_seconds, max_response_bytes, and tasks. HTTPS is required except explicitly enabled localhost HTTP. Credentials/query strings/fragments in URLs are rejected, redirects and environment proxy inheritance are disabled. The remote endpoint receives a versioned TaskRequest JSON and returns just the task's output object, which is schema-validated locally.

The destination is fixed by the operator and cannot be changed by a task. The operator must apply DNS/egress policy and endpoint trust; OFlow does not promise a complete network sandbox or a universal SSRF filter. The sample local inference server runs in [examples/external_inference.py](examples/external_inference.py).

## Errors and cancellation

Return only JSON-compatible, schema-valid outputs. Raise normal exceptions for per-request failure; public clients receive a stable category without traceback or exception text. Raise WorkerCrashed when the runtime must be recreated. Call context.check_cancelled() between meaningful units. A batch must preserve input association and handle cancellation per item, or document that a batch failure affects its remaining items.

Streaming capability declarations are rejected during registry validation because the beta transport is not operational. POST /v1/tasks/stream explicitly reports unsupported capability.
