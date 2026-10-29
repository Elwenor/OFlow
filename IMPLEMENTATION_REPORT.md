# OFlow 1.0-beta: current implementation report

Updated 29 October 2026 for the follow-up removal of experimental compatibility code. Package version remains 1.0.0b1; public API/config/schema version remains "1". Initial implementation and its original audit are retained in Git history through PR #1.

## Current scope

OFlow targets one physical host with CPU, CPU + one CUDA GPU, or CPU + multiple CUDA GPUs. The shared SDK/API scheduler is independent of model families and hardware-specific backend implementations. Portable CPU CI is a validation environment, not the exclusive deployment target.

The standalone experimental archive, oflow.legacy package, model-specific semantic geometry/selector/checkpoint adapter, related configuration and tests have been removed. There is no built-in ORIS/mmBERT compatibility path. General bounded batching, trusted plugins, execution profiles, resource leases, routing, worker supervision, tenant controls and optional PyTorch execution remain in place.

## Implemented runtime support

| Component | Current behavior |
|---|---|
| CPU/mock backends | Lexical diversity, numeric statistics and deterministic test workloads |
| HTTP backend | Fixed trusted endpoint, bounded responses and schema validation |
| torch_embed | Persistent deterministic random byte-embedding weights and input tensors on configured cpu/cuda:N; normalized results on CPU |
| Single GPU | configs/gpu.yaml and examples/gpu.py; SDK and authenticated HTTP use the same worker |
| Multiple GPUs | configs/multi_gpu.yaml and a concurrent example; independent explicitly placed workers |
| Resource controls | CPU/device concurrency, persistent/per-invocation memory budgets, unavailable-worker handling and safe lease lifetimes |
| Missing GPU | Requested embedding fails with unavailable; separately configured CPU tasks remain available; no silent substitution |

The included torch weights are demonstrative and untrained. Production models must be implemented as trusted backends or provided by a configured external service. Tensor parallelism, multi-node execution and built-in non-CUDA accelerator discovery are not implemented.

## Verification for this change

- Full CPU/API suite without PyTorch: **66 passed, 3 skipped**.
- Optional PyTorch suite on the existing CPU runtime: **3 passed, 2 skipped**. The skipped cases explicitly require one/two physical CUDA GPUs.
- Ruff lint and formatting passed; mypy passed across **16** package source files.
- Tests verify GPU config schemas, absent-device behavior through both SDK and HTTP, continued CPU task availability and existing mocked multi-GPU worker execution.
- New physical-CUDA tests verify CPU/CUDA output equivalence, persistent weight reuse, repeated allocation cleanup, actual two-device placement and final lease cleanup when run on suitable hardware.
- A clean wheel build/install is checked to exclude all removed module files. Runtime tests and source/package checks are distinct from hardware validation.
- Synthetic scheduler benchmark is refreshed for the current source tree, with one warmup and three measured repetitions per scenario. It measures CPU scheduler behavior, not GPU/model performance.

The local machine has Windows 11, Python 3.12.2, 16 logical CPU threads, approximately 31.92 GiB RAM and PyTorch 2.14.1+cpu. No physical CUDA device is available. A skipped test is not evidence of working physical GPU execution. GitHub Actions validates the CPU matrix (Linux/Windows, Python 3.11-3.13) and a separate torch CPU job; consult the actual PR checks for the current result.

## Remaining limits

Real CUDA execution, VRAM behavior and performance must be validated on the target host. Docker runtime validation is still pending because the local Docker daemon is unavailable. Trusted plugins are in-process Python without isolation. Memory reservations are estimates; cancellation cannot forcibly kill native code or kernels, and shutdown waits for physical completion. Metadata audit does not provide durable job recovery. Functional streaming, managed workflows, distributed execution and tensor parallelism remain future work.

See GPU.md for exact device/runtime support and hardware acceptance commands, MIGRATION.md for the intentional removal of the experimental surface, and the architecture/security/deployment documentation for operational boundaries.
