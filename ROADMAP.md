# Roadmap

## Implemented beta foundation

Versioned task contracts; strict YAML; shared SDK/API services; bounded queues/physical workers; tenant quotas; deterministic ordered routing; profiles/microbatching; cooperative cancellation/deadlines; resource leases and device inventory; trusted plugins; CPU/mock/HTTP/optional torch backends; metadata audit, logs, Prometheus; CLI, examples, packaging and configured CPU CI.

## Next stage: hardware and production-model acceptance

1. Run optional CUDA tests on one and multiple physical GPUs, measuring allocated/reserved VRAM and pinned-memory high-water marks during long workloads and failures.
2. Supply original ORIS/mmBERT selector/pipeline/checkpoints; generate independent reference fixtures and gate labels, geometry, truncation and precision.
3. Measure real latency/throughput/queue tradeoffs across candidate batch/token-window/padding settings with repeated, randomized, correctness-gated runs.
4. Validate Docker build/runtime and full configured CI matrix; pin tested dependency sets for supported deployments.
5. Add explicit process-isolated workers and hard termination before serving untrusted workloads or requiring bounded shutdown despite hung native code.

## Later, only when justified

Functional streaming backends/transports; remote-worker transport; managed workflows; durable jobs and crash recovery; optional PostgreSQL; optional ONNX/vLLM adapters; richer metrics/tracing exporters; operational key rotation and rate limiting.

Multi-node scheduling, Ray/Kubernetes, tensor parallelism, plugin marketplace, billing, training and graphical workflow editing are not implemented or required by this beta. They must not become mandatory dependencies of local CPU usage.
