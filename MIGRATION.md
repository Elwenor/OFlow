# Legacy migration and audit

The supplied 2,880-line experimental file was audited across v3 launch/prefetch, semantic geometry, v4 exact batching, Kania changes, CLI and phased autotuning. The original GitHub repository contained README and Apache-2.0 LICENSE only. Neither selector/pipeline sources nor model/tokenizer assets were present.

The original file is preserved byte-for-byte as `legacy/oflow_v5_beta.original.py`, SHA-256 **d010b66f6c5af941dc6ca37bea842a749e08ee5a548ff4ab1dca11e93b7eaa26**. Its v3/v4/Kania symbols and original experimental CLI remain available for reproduction in a trusted original environment. This archive is not the server implementation and retains its original risks.

New code separates scheduler/resource management from optional legacy inference. Semantic geometry was extracted into oflow.legacy.semantic with formatting/name changes, keeping model-specific thresholds/schema outside Core. The new adapter loads one persistent model/tokenizer, caps documents/paragraphs, preserves output association, and uses bounded tokenizer/semantic pipelines. The original fast-head shortcut, SDPA tuning and compile experiments remain in the archive; the new adapter deliberately executes the model's ordinary forward path until independent model equivalence is demonstrated.

## Fourteen investigated risks

Original line references refer to the unchanged archive.

| Risk | Evidence in original | New path / disposition |
|---|---|---|
| 1. Retained GPU outputs | _gpu_launch_batches and _v4_gpu_exact append every device output, then concatenate (lines 468–576, 2028–2167) | Blocking per-batch D2H; only CPU outputs accumulate within bounded task size |
| 2. Retained pinned buffers | live_cpu_inputs/live_cpu retained until stage-wide synchronization | Maximum two staging slots, released only after required stream/copy completion |
| 3. Retained GPU inputs | live_gpu_inputs/live_gpu keep all input batches | Per-slot lifetime plus record_stream; no all-input GPU list |
| 4. Unbounded semantic futures | v3 executor.map eagerly submits jobs; v4 semantic_futures grows per chunk (2372–2425) | bounded_map strictly caps submitted semantic futures |
| 5. Window memory | v4 materializes all windows; Kania forces embedding lookahead >=8192 | Lazy window iteration, configurable bounded window/depth; no forced large window |
| 6. Async lifetimes | Original conservatively retains buffers and uses events/record_stream; corruption is not confirmed | Explicit source ownership, event wait, record_stream and synchronization in failure/finally paths; physical CUDA test pending |
| 7. Global torch settings | oflow_attach_max changes matmul precision/TF32/cuDNN globally (2280–2283) | New adapter makes no global tuning changes; external trusted loaders still require their own audit |
| 8. Equivalence | Original checks only final repetition, absolute tolerance and semantic subsets | New comparator checks full structure/labels/finite floats; every candidate repetition is gated; analytic geometry tests plus archive differential test; actual selector fixture test provided |
| 9. Invalid winners | v3 falls back to non-equivalent candidates (1503–1511); mixed SDPA appended without equivalence filter (2736–2745) | Generic autotuner cannot select any candidate failing a gate; archive left unchanged for provenance |
| 10. Repetitions/order | v4 defaults to 2 and deterministic phase order | New benchmarks require >=3 measured runs plus warmup and randomized order |
| 11. Deserialization | _resolve_runtime explicitly uses weights_only=False (366) | New probe uses weights_only=True and requires maintained torch>=2.10; all checkpoints and external loaders remain operator-trusted |
| 12. Hardcoded paths | Parser defaults reference specific model runs/corpus | New plugin requires explicit operator asset paths; no model path in Core or task requests |
| 13. Validation | Some candidate values validated; numerous v4 numeric flags silently clamped/unvalidated | Strict validated configuration, explicit limits, budget/divisibility checks and oversized-item rejection |
| 14. Hidden dependencies | Dynamic selector/pipeline import, load_mmbert hooks, PersistentTorchPatch | Documented explicit trusted assets; no PersistentTorchPatch in new adapter; missing dependencies fail clearly |

An archive differential test proves extraction stability, not independent numerical equivalence to the missing production selector. The analytic tests provide independent small-case checks. New CUDA ownership follows [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda); checkpoint restrictions follow [serialization guidance](https://docs.pytorch.org/docs/stable/notes/serialization). Neither source establishes physical validation on this machine.

## Use the optional adapter

Install a suitable maintained PyTorch build plus dependencies of your original code. Edit configs/legacy.example.yaml with explicit trusted selector, ORIS pipeline (not needed for mmBERT), checkpoint and tokenizer paths. The allowlisted oflow.legacy plugin registers legacy.semantic. Set model_family appropriately; use device and memory budgets for CUDA. The trust_external_code flag acknowledges that those Python files and their loaders execute with host-process privileges. No automatic downloads or conversion of unsafe checkpoint formats occurs.

```python
async with OFlowEngine.from_config("configs/legacy.local.yaml") as engine:
    result = await engine.execute(TaskRequest(task="legacy.semantic", input={"texts": ["Document text"]}))
```

Reference regression: create a JSON fixture `{"texts": [...], "output": {"documents": [...]}}` from the original selector. Set OFLOW_LEGACY_CONFIG to your working adapter YAML and OFLOW_LEGACY_REFERENCE_JSON to that fixture, then run `python -m pytest -m legacy`. The output comparison must pass before claiming model compatibility. Empty paragraphs, label changes, ties, dtype/precision, truncation and model-specific head behavior need representative fixtures.

Missing source assets, actual-model equivalence, long CUDA runs, multi-GPU native validation and legacy performance improvements remain **EXPERIMENTAL / NOT VALIDATED**. No legacy throughput gain is claimed.
