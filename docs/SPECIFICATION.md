# OFLOW 1.0-BETA — FINAL ENGINEERING SPECIFICATION

> Historical design input. The subsequent owner decision removes the experimental source archive, model-specific compatibility adapter and associated tests/configuration. Those migration requirements below are superseded; CPU, single-GPU and multi-GPU remain supported deployment targets. Current behavior is documented in README.md and GPU.md.

## ROLE AND MISSION

You are the lead architect and implementation engineer responsible for transforming the provided OFlow experimental code into OFlow 1.0-beta.

Act as an expert in:

- Python systems architecture
- asynchronous execution
- inference serving
- CPU and GPU scheduling
- CUDA memory management
- multi-user API design
- plugin architecture
- performance benchmarking
- security engineering
- automated testing
- production software maintenance

Your responsibility is to implement a working, extensible, tested foundation, not merely describe one.

You have access to the existing repository and the supplied OFlow source file.

Inspect everything relevant before modifying the code.

Preserve existing functionality and user changes.

Do not fabricate completed features or successful tests.

Do not make destructive repository changes.

---

# 1. APPROVED PRODUCT DECISIONS

The project owner has approved the following architectural decisions. Treat them as binding requirements.

## Infrastructure

OFlow 1.0-beta targets ONE PHYSICAL MACHINE.

It must support:

- CPU-only execution
- CPU + single GPU
- CPU + multiple GPUs
- multiple backends
- multiple concurrent users
- multiple execution profiles

It must remain architecturally extensible to future multi-node execution.

DO NOT implement a mandatory distributed cluster for 1.0-beta.

DO NOT require Kubernetes, Ray, Redis, NATS or a distributed database for normal operation.

## Interfaces

Both interfaces are mandatory:

1. Direct local Python SDK.
2. HTTP API.

Both must use the same application services and execution Core.

Local SDK execution must not internally require HTTP communication.

## Plugins

Only trusted operator-installed plugins are permitted.

Remote API users cannot install plugins, load arbitrary Python modules or execute arbitrary code.

Dynamic plugin registration is supported only through explicit trusted configuration or operator-controlled installation.

Do not claim in-process plugins are securely isolated from one another.

## Product purpose

OFlow is a GENERAL AI WORKLOAD EXECUTION PLATFORM.

It must never become hardcoded to ORIS, Kania, BERT, mmBERT, scoring or any other single model family.

---

# 2. PRODUCT DEFINITION

OFlow manages AI tasks through configurable backends and execution policies.

The platform must support different use cases without requiring modifications to the central scheduler.

Examples:

- bulk document scoring;
- interactive text classification;
- semantic embeddings;
- translation;
- inference routing;
- reasoning-mode selection;
- remote model execution;
- future predictive workloads;
- future custom AI processing tasks.

Do not implement every possible AI task in the beta.

Implement a solid general contract and enough real examples to prove extensibility.

Model-independent execution is mandatory.

---

# 3. CORE ARCHITECTURAL PRINCIPLE

Separate six fundamental concepts:

TASK:
What computation is requested?

BACKEND:
Which registered implementation provides the capability?

RUNTIME:
How is the backend executed?

POLICY:
How are routing and scheduling decisions made?

PROFILE:
Which execution optimization objectives apply?

WORKFLOW:
How are multiple tasks composed?

These concepts must not be conflated.

A backend may implement multiple tasks.

A task may have multiple compatible backends.

An execution profile must not be tied to a specific model.

A routing policy must not be hardcoded into the scheduler.

---

# 4. ARCHITECTURAL BOUNDARIES

Implement clear boundaries for:

1. Core contracts
2. Application services
3. Task registry
4. Backend registry
5. Routing policies
6. Execution scheduler
7. Resource manager
8. Worker supervisor
9. Backend runtimes
10. Configuration
11. Local SDK
12. HTTP API
13. Authentication and tenant policy
14. Telemetry
15. Benchmarking
16. Legacy compatibility

Prefer simple cohesive modules.

Do not create a large number of empty abstraction files.

Avoid unnecessary inheritance hierarchies.

Use Python Protocols, dataclasses or validated models where appropriate.

Make dependency direction explicit.

Framework-specific code must not leak into pure Core contracts.

---

# 5. REQUIRED CORE CONTRACTS

Create typed, documented interfaces for:

- TaskRequest
- TaskResult
- TaskError
- TaskCapability
- BackendDescriptor
- BackendHealth
- ExecutionContext
- ExecutionProfile
- ExecutionPolicy
- ResourceRequirements
- ResourceLease
- WorkerDescriptor
- RoutingDecision

TaskRequest should include:

- request ID;
- task identifier;
- task schema version;
- input;
- allowed execution options;
- selected profile;
- optional deadline;
- priority;
- optional preferred backend;
- correlation ID.

Trusted tenant identity must be injected by the authentication layer.

Never trust tenant identity provided in the request body.

TaskResult should contain:

- request ID;
- status;
- normalized output;
- selected backend;
- backend version;
- execution timing;
- optional usage statistics;
- safe metadata;
- normalized error information.

Define stable error categories.

Do not return arbitrary exception tracebacks to API clients.

Version public contracts from the beginning.

---

# 6. EXECUTION PROFILES

Implement three configurable execution profiles.

## latency_first

Optimize interactive execution.

Consider:

- queue wait time;
- time to first result;
- p50 latency;
- p95 latency;
- p99 latency;
- deadline compliance;
- low batching delay.

Support limited configurable microbatching when the backend allows it.

## throughput_first

Optimize bulk workloads.

Consider:

- items per second;
- tokens per second;
- batching efficiency;
- GPU utilization;
- total processing time;
- memory stability.

Support token-budget batching through optional backend capabilities.

## balanced

Support mixed workloads.

Prevent throughput-oriented work from monopolizing resources required by interactive traffic.

These profiles must be configurable.

Do not implement three separate schedulers with duplicated logic.

Implement a shared scheduling framework with replaceable policies.

Document the tradeoffs of each policy.

---

# 7. LOCAL SCHEDULER

Build a bounded, asynchronous scheduler.

Mandatory capabilities:

- bounded request queues;
- admission control;
- priority handling;
- fairness;
- deadlines;
- cancellation;
- graceful shutdown;
- worker health;
- resource reservations;
- overload rejection;
- execution status tracking.

Do not launch unbounded threads.

Do not create unlimited futures.

Do not block the API event loop with CPU-intensive work or GPU inference.

Use controlled workers or executors.

Define a clear task state machine.

Required states:

PENDING
ADMITTED
RUNNING
COMPLETED
FAILED
CANCELLED
TIMED_OUT
REJECTED

Define valid transitions.

Handle worker crashes.

Ensure resource leases are released after failures.

Cancellation of GPU work may be cooperative or delayed.

Document those limitations accurately.

A client disconnect must not result in uncontrolled memory accumulation.

---

# 8. FAIRNESS AND MULTI-USER SUPPORT

OFlow may serve multiple users simultaneously.

Implement a trusted identity context.

Provide an initial API-key authentication mechanism suitable for controlled deployments.

Support:

- per-tenant concurrency limits;
- per-tenant queue limits;
- task permissions;
- backend permissions;
- configurable request priorities;
- overload rejection;
- administrative authorization.

Demonstrate concurrent requests from at least two tenants.

One tenant must not indefinitely monopolize execution capacity.

Bulk work must not permanently starve interactive work.

Do not invent complicated billing or subscription systems.

Authentication and quotas should be simple, documented and testable.

Use secure defaults for network-exposed operation.

---

# 9. HARDWARE RESOURCE MANAGER

The system must run without a GPU.

CPU-only operation is a mandatory acceptance criterion.

Detect available resources safely.

Where possible, inspect:

- CPU count;
- RAM;
- GPU count;
- GPU device names;
- VRAM;
- CUDA capabilities;
- runtime availability.

Avoid importing heavyweight frameworks during basic Core initialization.

Use lazy initialization and optional dependencies.

## Device allocation

Support logical device identifiers such as:

cpu
cuda:0
cuda:1

Do not assume GPU indices are stable across different host configurations.

Resolve and record actual device identity during startup.

Respect configured device visibility and restrictions.

Support placement of independent backend workers across devices.

Do not assign multiple workers to the same GPU without considering configured memory budgets and concurrency.

Use resource leases.

Prevent overcommit where resource estimates allow it.

Handle inaccurate memory estimates safely.

## Multi-GPU scope

For 1.0-beta implement:

- multi-device inventory;
- device-specific worker placement;
- independent workers per GPU;
- capability-aware backend allocation;
- resource usage accounting;
- device health reporting;
- graceful behavior when a GPU is unavailable.

If actual multi-GPU hardware is unavailable, validate placement logic with mocks and report that physical validation remains pending.

Do not implement custom tensor parallelism.

---

# 10. WORKER SUPERVISOR

Implement a local worker supervisor abstraction.

It should manage:

- worker startup;
- backend initialization;
- worker health;
- execution capacity;
- graceful shutdown;
- failure reporting;
- safe restart policies.

Prefer persistent workers for heavyweight backends.

Do not reload model weights per request.

Avoid repeatedly loading the same model unnecessarily.

Do not automatically duplicate large model weights across devices without explicit configuration.

Design the worker transport interface so that a future remote worker transport can be added.

For beta, only local execution transport needs to be operational.

Do not create fake distributed functionality.

---

# 11. BACKEND REGISTRY

Backends must declare their capabilities.

Examples:

text.classify
text.score
text.embed
text.translate
text.generate
route.select

Use namespaced task identifiers.

Require each backend to declare:

- stable ID;
- version;
- runtime;
- supported tasks;
- input schemas;
- output schemas;
- execution requirements;
- batching capabilities;
- streaming capabilities;
- concurrency limits;
- health behavior.

Provide lifecycle methods.

Suggested capabilities:

initialize
load
execute
execute_batch
stream
health
shutdown

Do not force implementations to support unsupported methods.

Use explicit optional capability contracts.

The registry must validate task/backend compatibility before execution.

---

# 12. TRUSTED PLUGIN SYSTEM

Plugins are installed only by the operator.

Use an explicit allowlist or trusted package registry.

Plugin metadata must identify:

- plugin name;
- version;
- supported API version;
- capabilities;
- entry point;
- dependency requirements.

Reject incompatible plugin API versions.

Do not accept arbitrary import paths through public API requests.

Do not download or execute plugins automatically from user-submitted URLs.

Document that trusted Python plugins can execute code with the privileges of their hosting process.

A future isolated plugin mode may be designed but is not mandatory.

Do not build a full plugin marketplace.

---

# 13. ROUTING ENGINE

Implement configuration-driven routing.

A routing decision should consider:

- task requirements;
- backend capabilities;
- backend health;
- tenant authorization;
- runtime availability;
- execution policy;
- optional deadline;
- resource availability.

The same task must be routable to different backends without changing Core code.

Provide a deterministic rules-based router.

Support ordered fallback rules.

Do not silently fallback to an incompatible backend.

Include an explainable RoutingDecision structure.

Example:

A request asks for a reasoning-capable response.

A registered policy examines compatible backends and chooses an allowed backend.

No model name or reasoning-specific special case should be hardcoded into the generic scheduler.

Advanced learned routing is a future capability, not a beta requirement.

---

# 14. LOCAL SDK

Provide a clean Python API.

Target usage:

```python
from oflow import OFlowEngine, TaskRequest

async with OFlowEngine.from_config("oflow.yaml") as engine:
    result = await engine.execute(
        TaskRequest(
            task="text.score",
            input={"text": "Example text"},
            profile="latency_first",
        )
    )
```

The actual API may be adjusted for type safety and ergonomics.

Provide synchronous convenience methods where appropriate.

Clearly define engine lifecycle and shutdown.

Local use must not require:

- an HTTP server;
- API-key authentication for the same-process caller;
- a database server;
- Ray;
- a message broker.

Local use must still respect configured resource limits.

---

# 15. HTTP API

Use FastAPI and ASGI.

Implement versioned endpoints.

Minimum endpoints:

GET /health/live
GET /health/ready
GET /v1/capabilities
GET /v1/backends
POST /v1/tasks/execute
POST /v1/tasks/batch

Implement a streaming endpoint if a genuinely functional streaming-capable example backend is available.

Otherwise provide the contract and explicit unsupported-capability behavior without claiming working streaming.

API requirements:

- request validation;
- authentication;
- authorization;
- bounded request size;
- timeouts;
- overload handling;
- stable errors;
- cancellation on disconnect where feasible;
- request IDs;
- health endpoints.

Protect backend inventory details where appropriate.

Separate public and administrative operations.

The HTTP API must use the same execution services as the SDK.

Do not duplicate scheduling logic.

---

# 16. LEGACY OFLOW MIGRATION

The supplied Python file contains:

- OFlow v3 functionality;
- OFlow v4 MAX functionality;
- Kania v2 batching improvements;
- CPU/GPU overlap;
- CUDA transfer optimizations;
- semantic postprocessing;
- benchmark and autotuning code.

Audit the complete file.

Do not transfer its architecture directly into Core.

Extract reusable mechanisms into suitable optional modules.

Legacy ORIS and mmBERT support must become adapter functionality.

Model-specific semantic geometry must remain outside generic Core.

Preserve old functionality through compatibility paths where possible.

If external source files or checkpoints are missing, do not fabricate compatibility.

Provide clear dependency requirements and tests that can run when the original environment is available.

---

# 17. FIX KNOWN LEGACY RISKS

Specifically investigate and correct:

1. Retained GPU outputs during long inference.
2. Retained pinned CPU buffers.
3. Retained GPU input buffers.
4. Unbounded semantic futures.
5. Excessive tokenization window memory.
6. Incorrect asynchronous transfer lifetimes.
7. Global PyTorch configuration side effects.
8. Weak numerical equivalence testing.
9. Selecting invalid benchmark winners.
10. Insufficient benchmark repetitions.
11. Unsafe checkpoint deserialization.
12. Hardcoded model-specific paths.
13. Weak parameter validation.
14. Hidden dependencies on selector and pipeline modules.

Do not assume these are all confirmed runtime bugs.

Verify each issue against the implementation and document the resolution.

Use bounded pipelines and explicit memory ownership.

Maintain correctness before pursuing performance improvements.

---

# 18. BATCHING STRATEGY

Batching must be optional and capability-driven.

Backends may support:

- no batching;
- fixed-size batching;
- dynamic batching;
- token-budget batching;
- shape bucketing;
- padding-aware batching.

Do not apply NLP-specific token batching to arbitrary workloads.

Introduce a batching strategy interface.

Support workload-specific cost estimates.

Use bounded lookahead.

Preserve input-to-output association even when batches are reordered.

For Kania-style padding-aware batching, make efficiency thresholds configurable.

Do not hardcode 88% as a universal optimum.

Test multiple candidate configurations through benchmarks.

---

# 19. AUTOTUNING AND BENCHMARKING

Keep benchmarking separate from normal request execution.

Autotuning must be explicitly requested.

Benchmark:

- latency;
- throughput;
- memory;
- queueing;
- concurrency;
- batching;
- deadline behavior.

Use warmups.

Use multiple measured repetitions.

Avoid always benchmarking candidates in identical order.

Record environment information.

Never claim an optimization is universally optimal.

Associate benchmark results with:

- hardware;
- backend;
- model version;
- runtime;
- workload;
- configuration.

Use correctness gates.

A candidate that violates required output equivalence must not win.

Do not apply benchmark configurations across incompatible environments without validation.

---

# 20. CONFIGURATION SYSTEM

Use Pydantic models and YAML configuration.

Support configuration schema versions.

Separate:

- general system configuration;
- device configuration;
- backend configuration;
- task registration;
- execution profiles;
- routing policies;
- tenant limits;
- API settings;
- telemetry settings.

Define configuration precedence.

Reject invalid critical fields.

Do not silently accept unsupported options.

Support safe application of selected runtime changes.

Not every setting must be hot-reloadable.

Changes requiring worker recreation or restart must be identified explicitly.

Provide configuration validation without loading GPU models.

---

# 21. PERSISTENCE

Use SQLite for optional local persistent metadata.

The Core must remain usable without a database for simple synchronous execution.

Separate persistent state from temporary execution state.

Do not store document contents or prompts by default.

Support optional audit records and job metadata.

Use a repository abstraction that can later support PostgreSQL.

Do not implement a mandatory distributed broker.

Do not claim durable execution unless job persistence and recovery are actually implemented.

Exactly-once execution is not a beta requirement.

Document delivery and retry semantics honestly.

---

# 22. OBSERVABILITY

Implement structured logs.

Track:

- request ID;
- task type;
- backend;
- execution profile;
- queue wait;
- execution time;
- status;
- error category;
- worker identity;
- tenant identity where safe.

Avoid recording sensitive model inputs.

Support OpenTelemetry instrumentation where practical.

Provide metrics suitable for Prometheus.

Track:

- active requests;
- queued requests;
- rejected requests;
- completed requests;
- failed requests;
- cancelled requests;
- deadline misses;
- backend health;
- worker state;
- resource usage;
- throughput;
- latency distributions.

Do not make observability dependencies mandatory for CPU-only library usage.

---

# 23. SECURITY REQUIREMENTS

For network operation:

- API authentication must be enabled by default.
- Administrative operations require explicit privileges.
- Tenant identity is established by trusted middleware.
- Request size and concurrency are limited.
- Arbitrary plugin installation is forbidden.
- Arbitrary file paths cannot be used to load code through normal task requests.
- External backend URLs must be operator-configured.
- SSRF protections must be considered.
- Secrets must not appear in logs.
- Input data must be validated.
- Unsafe deserialization must be reviewed.

Local trusted SDK operation may use a different authentication boundary, but must not silently grant remote administrative access.

Document deployment security assumptions.

---

# 24. OPTIONAL TECHNOLOGY ADAPTERS

The recommended stack is:

Core:
- Python 3.11+
- asyncio
- typed contracts
- Pydantic v2

HTTP:
- FastAPI
- ASGI server

Testing:
- pytest
- type checking
- linting

Model runtimes:
- optional PyTorch
- optional ONNX Runtime
- optional remote HTTP

Future integration:
- optional vLLM backend
- optional Ray execution adapter
- optional PostgreSQL
- optional durable queue

Do not make future components mandatory.

If a library is unnecessary for the initial implementation, leave it optional or defer it.

Confirm compatible dependency versions against the actual repository environment.

---

# 25. INITIAL FUNCTIONAL IMPLEMENTATION

Deliver a functioning vertical slice.

Minimum working components:

1. Core contracts.
2. Configuration loader.
3. Backend registry.
4. Deterministic mock backend.
5. CPU computation backend.
6. Local scheduler.
7. Resource manager.
8. Local SDK.
9. HTTP API.
10. Authentication.
11. Tenant-aware admission limits.
12. Routing policies.
13. Execution profiles.
14. Structured logging.
15. Tests.
16. CLI.
17. Examples.

Then integrate legacy OFlow where dependencies permit.

Do not create unfinished stubs masquerading as supported features.

---

# 26. REQUIRED EXAMPLES

Create runnable examples.

### Example 1 — CPU-only

Start OFlow without GPU dependencies.

Execute a local task.

### Example 2 — API

Start the server and execute an authenticated request.

### Example 3 — Low latency

Execute interactive requests using latency_first.

### Example 4 — Bulk scoring

Execute a collection of items using throughput_first.

### Example 5 — Routing

Change the selected backend through YAML configuration.

No changes to Core source code.

### Example 6 — Concurrent clients

Simulate multiple clients and demonstrate quotas.

### Example 7 — Multi-GPU placement

Show configuration and resource allocation across multiple GPU devices.

If real hardware is absent, use mocked resource inventory tests.

### Example 8 — External inference

Demonstrate a trusted, explicitly configured HTTP backend using a local test server.

---

# 27. TESTING REQUIREMENTS

Implement automated tests covering:

- CPU-only startup;
- missing GPU support;
- backend registration;
- task compatibility;
- configuration validation;
- routing changes;
- admission control;
- queue bounds;
- cancellation;
- deadlines;
- worker failures;
- resource cleanup;
- concurrent tenants;
- authentication;
- authorization;
- API errors;
- memory growth;
- scheduler fairness;
- graceful shutdown.

Test repeated request execution.

Test with deliberately slow backends.

Test overload conditions.

Test that a bulk workload cannot cause permanent starvation.

Test that an unauthorized tenant cannot access restricted tasks.

Include optional CUDA tests.

Do not require GPU hardware for standard CI.

---

# 28. PERFORMANCE VALIDATION

Measure latency and throughput separately.

Do not use one metric as a substitute for the other.

Test scenarios:

A. One interactive user.

B. Multiple interactive users.

C. Large bulk workload.

D. Mixed interactive and bulk workload.

E. Backend failures under load.

F. Deadline expiration.

G. Memory pressure.

Use synthetic deterministic backends for portable scheduler testing.

Use actual models only when available.

Report measured numbers, environment, sample sizes and limitations.

Never invent benchmark results.

---

# 29. DEPLOYMENT

Support:

- editable development installation;
- packaged Python installation;
- local CLI;
- local HTTP server;
- Docker image;
- Docker Compose example.

CPU deployment must not require NVIDIA tooling.

GPU deployment should document compatible drivers, runtimes and dependencies.

Use environment-based secret configuration.

Provide startup and shutdown procedures.

Do not introduce Kubernetes in beta.

The architecture may describe future remote-worker transport, but do not implement a distributed cluster unless specifically authorized later.

---

# 30. DOCUMENTATION

Required documentation:

README.md
ARCHITECTURE.md
CONFIGURATION.md
BACKENDS.md
API.md
SECURITY.md
DEPLOYMENT.md
BENCHMARKING.md
MIGRATION.md
ROADMAP.md

Document what works today.

Separate experimental functionality from stable behavior.

Provide complete examples.

Explain how to create a new task and backend without editing Core.

Explain local CPU deployment.

Explain GPU device configuration.

Explain multi-user limits.

Explain how to change latency/throughput profiles.

Explain limitations of the beta.

---

# 31. IMPLEMENTATION SEQUENCE

Work in controlled phases.

## Phase A — Audit

Inspect repository and legacy code.

Identify dependencies, risks and migration boundaries.

## Phase B — Foundation

Implement Core contracts, configuration and registry.

## Phase C — Execution

Implement scheduler, admission control, resource manager and workers.

## Phase D — Interfaces

Implement SDK and HTTP API on shared services.

## Phase E — Extensibility

Implement routing policies, profiles and trusted plugin registration.

## Phase F — Legacy integration

Wrap and progressively extract existing OFlow algorithms.

## Phase G — Validation

Implement tests, benchmarks, examples and deployment documentation.

Do not wait until the end to run tests.

Validate after each meaningful implementation stage.

---

# 32. ACCEPTANCE CRITERIA

OFlow 1.0-beta is acceptable when all the following are demonstrated:

[ ] CPU-only installation works.

[ ] Core can execute a task without HTTP.

[ ] HTTP API executes through the same Core.

[ ] A backend can be added without changing scheduler code.

[ ] A task can be routed to different compatible backends via configuration.

[ ] Latency and throughput profiles work.

[ ] Queues are bounded.

[ ] Deadlines are respected as far as the backend contract permits.

[ ] Cancellation cleans up request state and resources.

[ ] Multiple authenticated clients can execute concurrently.

[ ] Tenant limits are enforced.

[ ] Resource allocation supports multiple logical GPUs.

[ ] Missing GPU hardware does not break CPU execution.

[ ] Legacy functionality has a documented compatibility path.

[ ] Semantic regressions are tested where reference dependencies exist.

[ ] API security controls are tested.

[ ] Examples run as documented.

[ ] CI tests pass in the supported CPU environment.

[ ] GPU tests are reported accurately.

[ ] Documentation matches functionality.

[ ] Unimplemented distributed capabilities are explicitly identified as future work.

---

# 33. SCOPE CONTROL

The goal is a coherent beta, not an entire cloud platform.

Do not implement:

- mandatory multi-node clustering;
- Kubernetes orchestration;
- a plugin marketplace;
- execution of untrusted plugins;
- a custom GPU tensor-parallel runtime;
- a billing platform;
- a graphical workflow editor;
- an entire model training framework;
- every possible AI workload.

Focus on a correct, maintainable execution foundation.

Do not overabstract.

Do not create empty scaffolding.

Do not silently weaken acceptance criteria.

If time or environment constraints prevent completion, prioritize a verified working vertical slice and clearly enumerate remaining items.

---

# 34. EXPECTED FINAL REPORT

At completion provide:

1. Architecture decisions made.
2. Existing code preserved.
3. New components implemented.
4. Files modified.
5. Tests executed.
6. Test results.
7. Hardware tested.
8. API endpoints implemented.
9. Supported backends.
10. Working examples.
11. Security limitations.
12. Performance results, if measured.
13. Remaining beta limitations.
14. Recommended next development stage.

Clearly distinguish:

IMPLEMENTED
TESTED
EXPERIMENTAL
NOT IMPLEMENTED

Never mark a feature completed solely because its interface exists.

---

# 35. FINAL DIRECTIVE

Transform the supplied experimental OFlow code into a genuine OFlow 1.0-beta foundation.

The system must be:

- backend-agnostic;
- task-extensible;
- CPU-compatible;
- GPU-aware;
- multi-GPU-ready on one host;
- concurrency-safe;
- multi-user-capable;
- configuration-driven;
- measurable;
- secure by default for network use;
- accessible via local SDK and HTTP API;
- prepared for future distributed execution without requiring it today.

Preserve valuable legacy algorithms.

Keep the Core independent of particular model families.

Build a functional, tested implementation.

Start with the repository audit, then proceed with incremental implementation and verification.

Do not stop after producing an architectural document.
