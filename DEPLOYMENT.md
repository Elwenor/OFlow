# Deployment

## CPU development and package installation

Python 3.11–3.13 is the configured CI range. The implementation has been tested locally on Python 3.12.2/Windows 11. Base installation requires Pydantic v2 and PyYAML only:

```bash
python -m venv .venv
# activate the virtual environment using your shell
python -m pip install -e .
oflow validate --config configs/cpu.yaml
oflow run --task text.score --input '{"text":"example"}'
```

On Windows PowerShell, JSON argument quoting may differ by PowerShell version; the Python examples avoid shell-quoting differences.

Build/install a distributable wheel:

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip install dist/oflow_engine-1.0.0b1-py3-none-any.whl
```

Extras: api (FastAPI/Uvicorn), http (HTTP backend/client), hardware (psutil RAM discovery), torch (optional PyTorch), dev (pytest/lint/type checking). CPU users never need CUDA/NVIDIA tooling.

## HTTP startup and shutdown

```bash
python -m pip install -e '.[api,http]'
# Set OFLOW_ALPHA_API_KEY and OFLOW_BETA_API_KEY to distinct generated secrets.
oflow serve --config configs/api.yaml
```

The CLI starts one ASGI process. SIGINT/Ctrl-C or SIGTERM invokes application shutdown: stop admission, drain for shutdown_grace_seconds, signal remaining work, wait for physical completion, then shut down persistent backends and audit. A hung/noncooperative backend can extend shutdown beyond the grace interval; an external process manager may enforce a final termination timeout.

Alternatively set OFLOW_CONFIG and run `uvicorn oflow.api:app_from_env --factory --host 127.0.0.1 --workers 1`. Do not increase workers to scale GPU serving: each process gets separate models/quotas. Configure TLS/reverse proxy/body-header timeouts and firewall for public access. OFlow's application body timeout does not replace server/proxy header limits.

## Docker Compose

```bash
# Create .env from .env.example and fill both values with random secrets.
docker compose up --build -d
docker compose logs -f
docker compose down
```

The image runs as nonroot, publishes to host loopback, drops capabilities, enables no-new-privileges, uses a read-only filesystem and checks aggregate readiness. It contains no GPU dependency. Secrets are environment variables in this simple example; use a deployment secret manager where appropriate. The local Docker engine was unavailable during implementation, so image/Compose runtime validation remains pending.

## GPU and multiple GPUs

Install a maintained PyTorch build matching the target CUDA runtime/driver using the [official installation guide](https://pytorch.org/get-started/locally/). Do not assume the optional extra selects an appropriate accelerator wheel for every host. Run `oflow devices --cuda` to inspect visible devices. Set CUDA_VISIBLE_DEVICES before process startup and configure budgets by the visible logical IDs, recording resolved UUIDs from administrative inventory.

Use `configs/gpu.yaml` for a single GPU and `configs/multi_gpu.yaml` for independent workers on two GPUs. Run `python examples/gpu.py` or `python examples/multi_gpu.py`; both report failure when a requested GPU cannot execute. These configurations include a separate CPU backend for CPU tasks. The included torch byte embeddings are demonstrative random weights. GPU memory estimates must include weights, runtime overhead and the largest configured batch. There is no custom tensor parallelism or multi-node execution. See [GPU.md](GPU.md) for the exact support and validation matrix and hardware acceptance commands.

Only mocked placement and CPU torch operations were validated on the implementation machine. The optional CUDA tests must run on real hardware before claiming physical multi-GPU readiness.
