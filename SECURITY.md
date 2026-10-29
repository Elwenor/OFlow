# Security and trust boundaries

Network execution requires configured API keys by default; startup fails with missing, short or duplicate keys. Tenant identity and admin privileges come from operator config plus authentication, never the request body. Quotas, task/backend authorization and maximum priority apply to the shared execution service. Keys are compared as constant-size SHA-256 digests using constant-time comparison, are not stored in YAML and are never logged by OFlow.

The local SDK is a trusted same-process boundary. A local Python caller can configure plugins and inject identity and therefore already has operator-level code execution privileges. Do not wrap this administrative surface directly in another public API.

## Implemented controls

- Strict versioned schemas with unknown-field rejection; bounded HTTP body and item count.
- Bounded queue, concurrent HTTP calls, physical backend invocations, result history and adapter prefetch.
- Per-tenant concurrency, queued work, task/backend permissions, priority and explicit administrative authorization.
- Generic client errors without tracebacks, exception strings, prompts or reflected validation values.
- Logs/audit contain IDs, task/backend/profile, timings, status and error category, not inputs/outputs or keys.
- No HTTP plugin installation, arbitrary module/file loader, URL selection, automatic download or checkpoint selection.
- Fixed operator-configured HTTP destinations, bounded responses, disabled redirects/environment proxies, HTTPS by default.
- No built-in checkpoint/file loader is exposed. Production backend plugins own model loading and must use trusted artifacts and maintained runtimes.

## Required deployment assumptions

Use TLS at a trusted reverse proxy for traffic beyond loopback, a firewall, operator-managed egress rules and filesystem permissions, and rotate distinct API keys by restarting with updated environment values. Bind to loopback unless network access is deliberately configured. Keep the process, installed packages, model assets and plugin distribution trustworthy. Docs are disabled by default; health endpoints reveal only aggregate booleans.

The beta API-key mechanism has no built-in key expiry, remote key rotation, per-IP rate limiting, OAuth, user enrollment or distributed quota coordination. Use one ASGI process; additional processes create independent limits and independently loaded models. Admin inventory/metrics expose operational details only to explicitly privileged tenants.

## Limitations

Trusted Python plugins run in process with its full privileges. A thread is not a sandbox. Segmentation faults, process termination, hanging native code, hard CPU/RAM/VRAM isolation and protection between malicious plugins require external process/container controls. Memory reservations are estimates; external GPU allocations and inaccurate batch budgets can still cause OOM. Cancellation cannot kill arbitrary code or a CUDA kernel safely.

The standalone experimental script and its model-specific compatibility adapter are removed from the current package. Any external plugin supplying model/checkpoint loading remains trusted code and must be audited by its operator. See [MIGRATION.md](MIGRATION.md).

Even restricted checkpoint loading has limitations; follow [PyTorch serialization guidance](https://docs.pytorch.org/docs/stable/notes/serialization) and keep the runtime patched. API keys are not plugin or checkpoint trust signatures.

SQLite audit contains no prompts by design, but IDs and task names can still be sensitive. Set retention and file permissions appropriately. Result history holds outputs in memory; disable terminal retention when needed. Audit records may be dropped under backpressure (oflow_audit_dropped_total); they are not a compliance-grade append-only log.

Report vulnerabilities privately to the repository owner instead of publishing credentials or sensitive payloads in an issue.
