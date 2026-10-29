# Migration

The repository now contains the general OFlow execution platform. The original standalone experiment, model-specific compatibility package, selector/pipeline loader, semantic geometry, example configuration and associated tests are removed from the current tree and wheel.

This intentionally removes the experimental compatibility surface:

- oflow.legacy imports and its plugin entry point are no longer available.
- The legacy.semantic task is no longer registered.
- Model-specific checkpoint/selector/tokenizer paths have no built-in loader.
- Old configurations referring to the removed plugin must be replaced; they do not silently run another backend.

Earlier experiments and the initial audit remain in Git history through [PR #1](https://github.com/Elwenor/OFlow/pull/1). They are not part of the installed package. The historical engineering input in docs/SPECIFICATION.md is marked as superseded for those migration requirements.

Use TaskRequest with a registered task and a configured backend. Existing general SDK/API contracts, profiles, quotas, resource management, trusted plugin registration and fixed-URL HTTP inference remain available. Supply a production model through its own trusted backend, register its typed task schemas and declare supported devices, batching and memory requirements. See BACKENDS.md for the extension contract.

For accelerator deployments use configs/gpu.yaml or configs/multi_gpu.yaml and see GPU.md. Removing model-specific experiments does not remove CUDA execution or multi-device placement. The bundled torch backend remains a random-weight runtime demonstration; this change does not introduce a pretrained production model.
