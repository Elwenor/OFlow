# OFlow 1.0-beta

A Python execution engine for configurable AI workloads on **one machine**. Use the local SDK or an authenticated HTTP API; both call the same bounded, tenant-aware scheduler. Models and task semantics belong in backends, not the scheduler.

**Target:** CPU, CPU + one CUDA GPU, or CPU + multiple CUDA GPUs on one host. Device placement is selected in YAML; SDK and HTTP use the same execution engine. CPU is the portable test environment, not a restriction on deployment.

**Status:** working beta foundation. CPU execution and mocked multi-device scheduling are tested. The optional PyTorch backend contains a real CUDA execution path, but physical CUDA validation requires a GPU host. Its included byte embeddings use random demonstration weights; production models must be supplied by a trusted backend. See [GPU support](GPU.md) and the [implementation report](IMPLEMENTATION_REPORT.md).

## Quick start — CPU, no server or GPU required

Python 3.11+:

```bash
git clone https://github.com/Elwenor/OFlow.git
cd OFlow
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
oflow validate --config configs/cpu.yaml
python examples/cpu_only.py
```

```python
import asyncio
from oflow import OFlowEngine, TaskRequest

async def main():
    async with OFlowEngine.from_config("configs/cpu.yaml") as engine:
        result = await engine.execute(TaskRequest(
            task="text.score",
            input={"text": "Ala ma kota. Ala ma psa."},
            profile="latency_first",
        ))
        print(result.output)

asyncio.run(main())
```

The CPU `text.score` example computes **lexical diversity**. It is a transparent example workload, not a trained quality model. `math.statistics` demonstrates a non-text task. A deterministic mock and a fixed-URL HTTP backend are also included. Optional `torch_embed` runs real PyTorch byte-embedding operations with deterministic random weights; it is not a pretrained semantic model.

## HTTP API

```bash
python -m pip install -e '.[api,http]'
# Generate two distinct secrets, then set OFLOW_ALPHA_API_KEY and OFLOW_BETA_API_KEY.
python -c "import secrets; print(secrets.token_urlsafe(32))"
oflow serve --config configs/api.yaml
```

Set environment variables using your shell (`export NAME=value` on POSIX; `$env:NAME='value'` in PowerShell). Both keys must contain at least 32 characters. Alpha is an example administrator; Beta has restricted tasks, backends and quotas.

```python
import os, httpx
response = httpx.post("http://127.0.0.1:8000/v1/tasks/execute",
    headers={"Authorization": f"Bearer {os.environ['OFLOW_BETA_API_KEY']}"},
    json={"task": "text.score", "input": {"text": "Hello OFlow"}}, timeout=70)
response.raise_for_status()
print(response.json())
```

Read [API.md](API.md) for endpoints, errors, batch limits, status and cancellation. TLS termination and egress controls belong to the deployment. No anonymous execution or remote plugin installation exists.

## Examples and verification

```bash
python examples/profiles.py             # latency-first + bounded bulk execution
python examples/routing.py              # switch backend through YAML data
python examples/external_inference.py   # self-contained local HTTP inference server
python -m examples.custom_plugin        # add a task/backend without Core changes
python examples/api_clients.py          # two authenticated clients; server must be running
python examples/gpu.py                  # one CUDA GPU; configs/gpu.yaml
python examples/multi_gpu.py            # optional PyTorch/CUDA; configs/multi_gpu.yaml

python -m pip install -e '.[dev,http,hardware]'
python -m pytest -q
python -m ruff check src tests examples
python -m mypy -p oflow
oflow benchmark --items 128 --repeats 3 --output reports/local-benchmark.json
```

CI covers CPU Python 3.11–3.13 on Linux and Windows, with a separate optional-PyTorch CPU job. That matrix describes configured checks; consult the actual GitHub run for its result.

## Operational boundaries

- Queues, execution slots, HTTP requests, prefetch and retained result history are bounded.
- Cancellation and deadlines can return before a backend finishes. Its worker slot and resource lease remain occupied until physical completion. Shutdown waits for noncooperative threads/CUDA work.
- Backends are trusted code in this process. Resource budgets are estimates, not OS or CUDA isolation.
- Audit SQLite stores bounded metadata only. Execution is in memory, without durable recovery or automatic retries.
- Streaming returns an explicit unsupported-capability error. Distributed execution, tensor parallelism and a workflow engine are future work.
- Model-specific experimental code and compatibility adapters are removed. Earlier experiments remain available in Git history; new models integrate through the trusted backend contract.

## Documentation

[Architecture](ARCHITECTURE.md) · [GPU support](GPU.md) · [Configuration](CONFIGURATION.md) · [Backends and plugins](BACKENDS.md) · [API](API.md) · [Security](SECURITY.md) · [Deployment](DEPLOYMENT.md) · [Benchmarking](BENCHMARKING.md) · [Migration](MIGRATION.md) · [Roadmap](ROADMAP.md) · [Implementation report](IMPLEMENTATION_REPORT.md)

Apache-2.0. The original repository license is preserved.
