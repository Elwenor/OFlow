# CPU and GPU deployment targets

OFlow targets one host with CPU, CPU + one NVIDIA CUDA GPU, or CPU + multiple CUDA GPUs. Configuration selects the device for each backend worker. CPU-only tests establish portability; they do not make CPU the only supported target.

## What the current implementation does

| Capability | Implementation | Validation |
|---|---|---|
| CPU execution | Bounded persistent workers; CPU and optional torch workloads | Real CPU tests and CI |
| CUDA execution | torch_embed puts weights and input tensors on configured cuda:N; results return to CPU | Implemented; physical-GPU tests included, not run on this CPU-only host |
| Multiple GPUs | Independent explicitly configured workers, device identities, memory budgets and ordered routing | Scheduler/placement tested with injected devices; physical two-GPU test included |
| Production model inference | Trusted backend or fixed remote HTTP endpoint supplies the model/runtime | No built-in pretrained production model is supplied |
| Tensor/model parallelism | One model split across GPUs | Not implemented |
| Other accelerators | ROCm, MPS, DirectML, TPU, custom hardware discovery | Not implemented by the built-in device manager |

The bundled torch_embed backend performs real tensor operations, including CUDA execution when selected. Its byte embeddings have deterministic random weights and are a runtime demonstration, not a trained semantic embedding model. The cpu scoring/statistics backend remains CPU-specific. A scheduler cannot make an arbitrary backend GPU-capable without that backend supporting the runtime.

## Single GPU

Install a maintained CUDA-enabled PyTorch build matching your driver using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). The torch extra declares the dependency but cannot choose the correct accelerator wheel for every host. Then:

```bash
python -m pip install -e '.[torch]'
oflow devices --cuda
oflow validate --config configs/gpu.yaml
python examples/gpu.py
```

The YAML places embeddings-gpu0 on cuda:0 with explicit persistent and per-invocation memory budgets. text.embed has no CPU fallback in this config: if CUDA is missing or its worker cannot initialize, the request returns unavailable and the example exits unsuccessfully. CPU tasks remain available through the separately configured CPU backend.

The GPU configurations also include the same environment-based authentication as the API example. Set distinct OFLOW_ALPHA_API_KEY and OFLOW_BETA_API_KEY secrets (at least 32 characters), install the api extra, and run `oflow serve --config configs/gpu.yaml` or `oflow serve --config configs/multi_gpu.yaml`. HTTP uses exactly the same GPU workers as the SDK. The SDK itself does not require these secrets. configs/api.yaml is a CPU demonstration, not an automatic hardware-selection policy. Aggregate readiness can remain true because its CPU worker is healthy; inspect the task result or administrative inventory to verify the requested GPU.

## Multiple GPUs

```bash
oflow validate --config configs/multi_gpu.yaml
python examples/multi_gpu.py
```

The example submits to both configured GPUs concurrently. Each backend entry deliberately creates its own independent weights. CUDA_VISIBLE_DEVICES must be set before startup; cuda:0/cuda:1 are indices within that visible set. Administrative inventory records actual UUIDs where available. Respect budgets and account for the largest batch, model weights, runtime overhead and memory consumed by other processes. This is independent task placement, not automatic tensor parallelism.

## Hardware acceptance

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest tests/test_torch.py -m cuda -rs
```

One-GPU tests verify actual weight placement, CPU/CUDA output equivalence, persistent weight reuse and repeated allocation cleanup. The two-GPU test verifies execution on both physical devices and lease/memory cleanup. They skip explicitly when hardware/runtime prerequisites are absent. A skipped test is not evidence of successful GPU execution.

The implementation machine has PyTorch 2.14.1+cpu and no CUDA device, so only CPU operations, configuration validation, unavailable-device behavior and mocked multi-device scheduling have been validated locally. Run the CUDA acceptance suite on the target machine before claiming physical GPU validation or model performance. GPU memory budgets remain cooperative estimates, not hardware isolation.
