from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


# =============================================================================
# OFlow v3 GPU lab
# Lightweight diagnostic/autotune harness for encoder pipelines.
#
# Goals of v0:
#   1) measure the machine and the real pipeline, not just model.forward()
#   2) isolate CPU/tokenizer/GPU/transfer/semantic-postprocess costs
#   3) try a small set of safe overlap configs automatically
#   4) verify output equivalence against the reference path
#   5) produce one JSON report that tells us what to change next
#
# This file intentionally does NOT replace the production corpus writer yet.
# It is a clean test harness. Once a configuration wins reliably, the winning
# scheduling path can be installed into the production pipeline.
# =============================================================================


def import_file(name: str, path: Path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    sys.modules.pop(name, None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_path(value: str, here: Path) -> Path:
    p = Path(value)
    if p.exists():
        return p.resolve()
    q = (here / p)
    if q.exists():
        return q.resolve()
    return p.resolve()


def parse_args():
    p = argparse.ArgumentParser("OFlow v3-gpu — GPU kernel / shape autotune")

    p.add_argument("--input", default=r"data\madlad_stage0\raw\pl_clean_0000.jsonl.gz")
    p.add_argument("--engine-script", default="v2_candidate_selector.py")
    p.add_argument("--pipeline-script", default="v2_stage_v2_pipeline_oris.py")

    p.add_argument("--backend", choices=["oris", "mmbert"], default="oris")
    p.add_argument(
        "--checkpoint",
        default=r"runs\oris_vs_mmbert_v4_1024\oris_small_c\best_model.pt",
    )
    p.add_argument(
        "--tokenizer",
        default=r"runs\oris_vs_mmbert_v4_1024\oris_small_c\tokenizer",
    )

    p.add_argument("--sample-docs", type=int, default=2048)
    p.add_argument("--max-scan", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--min-chars", type=int, default=300)
    p.add_argument("--max-chars", type=int, default=100_000)
    p.add_argument("--ngram-n", type=int, default=8)
    p.add_argument("--near-dup-threshold", type=float, default=0.86)

    p.add_argument("--document-max-length", type=int, default=1024)
    p.add_argument("--paragraph-max-length", type=int, default=256)
    p.add_argument("--max-paragraphs-per-doc", type=int, default=96)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    p.add_argument(
        "--baseline-config",
        default="8x64",
        help="reference document_batch x paragraph_batch",
    )
    p.add_argument(
        "--candidates",
        default="8x64x1x2x1,8x64x2x4x1,8x64x4x4x1,8x64x2x8x1,16x64x2x4x1,16x128x2x4x1",
        help=(
            "OFlow candidates: docBatch x paraBatch x tokenizerWorkers x prefetchDepth x semanticWorkers. "
            "4-value legacy form is also accepted. Example: 8x64x2x4x4"
        ),
    )
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.add_argument("--warmup-docs", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gpu-modes", default="native,bucket,bucket_compile", help="native,bucket,bucket_compile")
    p.add_argument("--bucket-window", type=int, default=256, help="bounded lookahead used for length bucketing")

    p.add_argument("--full-probe", action="store_true", default=True)
    p.add_argument("--no-full-probe", dest="full_probe", action="store_false")
    p.add_argument("--probe-mb", type=int, default=128)
    p.add_argument("--gzip-probe-records", type=int, default=5000)

    p.add_argument("--output", default="oflow_v3_gpu_report.json")
    return p.parse_args()


@dataclass
class Candidate:
    document_batch: int
    paragraph_batch: int
    tokenizer_workers: int
    prefetch_depth: int
    semantic_workers: int = 1
    gpu_mode: str = "native"

    @property
    def label(self):
        return (
            f"doc={self.document_batch} para={self.paragraph_batch} "
            f"tok_workers={self.tokenizer_workers} prefetch={self.prefetch_depth} "
            f"sem_workers={self.semantic_workers} gpu={self.gpu_mode}"
        )


def parse_pair(text: str) -> tuple[int, int]:
    a, b = text.lower().split("x", 1)
    return int(a), int(b)


def parse_candidates(text: str) -> list[Candidate]:
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        xs = raw.lower().split("x")
        if len(xs) not in (4, 5):
            raise ValueError(f"Bad candidate {raw!r}; expected 8x64x1x4 or 8x64x1x4x4")
        vals = [int(x) for x in xs]
        if len(vals) == 4:
            vals.append(1)
        c = Candidate(*vals)
        if min(c.document_batch, c.paragraph_batch, c.tokenizer_workers, c.prefetch_depth, c.semantic_workers) <= 0:
            raise ValueError(f"Candidate values must be >0: {raw}")
        out.append(c)
    if not out:
        raise ValueError("No OFlow candidates")
    return out


# =============================================================================
# Machine probe
# =============================================================================


def _run_cmd(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return None


def basic_system_probe() -> dict[str, Any]:
    out: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical": os.cpu_count(),
    }

    try:
        import psutil
        out["cpu_physical"] = psutil.cpu_count(logical=False)
        vm = psutil.virtual_memory()
        out["ram_total_gib"] = vm.total / 1024**3
        out["ram_available_gib"] = vm.available / 1024**3
    except Exception:
        pass

    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(i)
            out["gpu"] = torch.cuda.get_device_name(i)
            out["vram_gib"] = props.total_memory / 1024**3
            out["cuda_capability"] = list(torch.cuda.get_device_capability(i))
            out["multi_processor_count"] = getattr(props, "multi_processor_count", None)
    except Exception as exc:
        out["torch_probe_error"] = repr(exc)

    smi = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current,power.limit",
        "--format=csv,noheader,nounits",
    ])
    if smi:
        out["nvidia_smi"] = smi.splitlines()[0]

    return out


def full_transfer_probe(mb: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"enabled": False, "reason": "cuda_unavailable"}

    n = int(mb * 1024 * 1024)
    device = torch.device("cuda")
    result: dict[str, Any] = {"enabled": True, "buffer_mb": mb}

    try:
        host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(n, dtype=torch.uint8, device=device)
        stream = torch.cuda.Stream()

        # warmup
        with torch.cuda.stream(stream):
            dev.copy_(host, non_blocking=True)
        stream.synchronize()

        def bench_h2d(reps=5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                for _ in range(reps):
                    dev.copy_(host, non_blocking=True)
                end.record(stream)
            end.synchronize()
            ms = start.elapsed_time(end)
            gb = (n * reps) / 1e9
            return gb / (ms / 1000.0)

        def bench_d2h(reps=5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                for _ in range(reps):
                    host.copy_(dev, non_blocking=True)
                end.record(stream)
            end.synchronize()
            ms = start.elapsed_time(end)
            gb = (n * reps) / 1e9
            return gb / (ms / 1000.0)

        result["h2d_gb_s"] = bench_h2d()
        result["d2h_gb_s"] = bench_d2h()
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def storage_probe(path: Path, mb: int, gzip_records: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    nbytes = int(mb * 1024 * 1024)

    try:
        t0 = time.perf_counter()
        read = 0
        with path.open("rb") as f:
            while read < nbytes:
                chunk = f.read(min(8 * 1024 * 1024, nbytes - read))
                if not chunk:
                    break
                read += len(chunk)
        dt = max(time.perf_counter() - t0, 1e-9)
        result["raw_read_mb_s"] = (read / 1024**2) / dt
        result["raw_read_mb"] = read / 1024**2
    except Exception as exc:
        result["raw_read_error"] = repr(exc)

    if path.name.lower().endswith(".gz"):
        try:
            t0 = time.perf_counter()
            records = 0
            unc = 0
            with gzip.open(path, "rb") as f:
                for raw in f:
                    unc += len(raw)
                    records += 1
                    if records >= gzip_records:
                        break
            dt = max(time.perf_counter() - t0, 1e-9)
            result["gzip_records_s"] = records / dt
            result["gzip_uncompressed_mb_s"] = (unc / 1024**2) / dt
            result["gzip_records"] = records
        except Exception as exc:
            result["gzip_error"] = repr(exc)
    return result


# =============================================================================
# Ordered CPU tokenization prefetch
# =============================================================================


def _tokenize_one(tokenizer, texts: list[str], max_length: int):
    t0 = time.perf_counter()
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
        verbose=False,
    )
    return enc, time.perf_counter() - t0


def prefetched_tokenization(
    tokenizer,
    text_batches: list[list[str]],
    max_length: int,
    workers: int,
    depth: int,
):
    workers = max(1, int(workers))
    depth = max(workers, int(depth))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oflow-tokenizer") as pool:
        futures: dict[int, Any] = {}
        submit_i = 0

        def fill():
            nonlocal submit_i
            while submit_i < len(text_batches) and len(futures) < depth:
                i = submit_i
                futures[i] = pool.submit(_tokenize_one, tokenizer, text_batches[i], max_length)
                submit_i += 1

        fill()
        for i in range(len(text_batches)):
            fut = futures.pop(i)
            enc, tok_s = fut.result()
            fill()
            yield i, enc, tok_s


# =============================================================================
# OFlow semantic adapter
# =============================================================================


def _resolve_runtime(engine, args):
    import torch

    checkpoint_path = Path(args.checkpoint)
    ckpt_probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = str(
        ckpt_probe.get("model_name")
        or ckpt_probe.get("args", {}).get("model")
        or "jhu-clsp/mmBERT-small"
    )

    candidates: list[str] = []
    if args.tokenizer:
        candidates.append(args.tokenizer)
    candidates.extend([
        str(checkpoint_path.parent / "tokenizer"),
        str(checkpoint_path.parent),
        model_name,
    ])
    tokenizer_source = None
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            if p.is_dir() and any(
                (p / name).exists()
                for name in ("tokenizer.json", "tokenizer_config.json", "config.json")
            ):
                tokenizer_source = cand
                break
        elif "/" in cand:
            tokenizer_source = cand
            break
    if tokenizer_source is None:
        tokenizer_source = model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, schema, train_args = engine.load_mmbert(
        checkpoint_path,
        tokenizer_source,
        device,
    )
    return checkpoint_path, tokenizer_source, device, model, tokenizer, schema, train_args


_COMPILED_MODEL_CACHE = {}

def _maybe_compile_model(model, enabled: bool):
    if not enabled:
        return model
    import torch
    key = id(model)
    cached = _COMPILED_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile unavailable")
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
    _COMPILED_MODEL_CACHE[key] = compiled
    return compiled

def _bounded_length_order(items, text_getter, window: int):
    """Sort only inside bounded lookahead windows; keeps production-like memory/latency."""
    if window <= 1:
        return list(items)
    out=[]
    for start in range(0, len(items), window):
        chunk=list(items[start:start+window])
        chunk.sort(key=lambda x: len(text_getter(x)))
        out.extend(chunk)
    return out

def _make_cpu_batches(sample, batch_size: int, getter):
    batches = []
    spans = []
    for start in range(0, len(sample), batch_size):
        chunk = sample[start:start + batch_size]
        batches.append([getter(x) for x in chunk])
        spans.append((start, start + len(chunk)))
    return batches, spans


def _gpu_launch_batches(
    *,
    model,
    tokenizer,
    text_batches: list[list[str]],
    max_length: int,
    tokenizer_workers: int,
    prefetch_depth: int,
    pin_memory: bool,
    device,
    use_amp: bool,
    amp_dtype,
    want: str,
):
    """
    Launch all batches without per-batch D2H synchronization.
    CPU tokenization runs ahead while CUDA executes the previous batch.
    A separate CUDA copy stream stages H2D where possible.
    """
    import torch

    gpu_outputs = []
    # Keep staged buffers alive until the async copy/compute queues are drained.
    # This also makes Windows/CUDA allocator behaviour deterministic during v0.
    live_cpu_inputs = []
    live_gpu_inputs = []
    token_counts: list[list[int]] = []
    real_tokens_total = 0
    padded_tokens_total = 0
    tokenize_sum = 0.0
    pin_sum = 0.0
    cpu_submit_sum = 0.0
    copy_events = []
    model_events = []

    if device.type == "cuda":
        compute_stream = torch.cuda.current_stream(device)
        copy_stream = torch.cuda.Stream(device=device)
    else:
        compute_stream = None
        copy_stream = None

    stage_start = time.perf_counter()

    for _, enc, tok_s in prefetched_tokenization(
        tokenizer,
        text_batches,
        max_length,
        tokenizer_workers,
        prefetch_depth,
    ):
        tokenize_sum += tok_s
        _counts = [int(x) for x in enc["attention_mask"].sum(dim=1).tolist()]
        token_counts.append(_counts)
        real_tokens_total += sum(_counts)
        padded_tokens_total += int(enc["input_ids"].numel())

        cpu_inputs = {
            k: v
            for k, v in enc.items()
            if k in {"input_ids", "attention_mask"}
        }

        if device.type == "cuda" and pin_memory:
            tpin = time.perf_counter()
            for k, v in list(cpu_inputs.items()):
                if not v.is_pinned():
                    cpu_inputs[k] = v.pin_memory()
            pin_sum += time.perf_counter() - tpin

        submit0 = time.perf_counter()
        if device.type == "cuda":
            c0 = torch.cuda.Event(enable_timing=True)
            c1 = torch.cuda.Event(enable_timing=True)
            ready = torch.cuda.Event()
            with torch.cuda.stream(copy_stream):
                c0.record(copy_stream)
                gpu_inputs = {
                    k: v.to(device, non_blocking=pin_memory)
                    for k, v in cpu_inputs.items()
                }
                c1.record(copy_stream)
                ready.record(copy_stream)
            copy_events.append((c0, c1))
            live_cpu_inputs.append(cpu_inputs)
            live_gpu_inputs.append(gpu_inputs)

            compute_stream.wait_event(ready)
            for _tensor in gpu_inputs.values():
                _tensor.record_stream(compute_stream)
            m0 = torch.cuda.Event(enable_timing=True)
            m1 = torch.cuda.Event(enable_timing=True)
            m0.record(compute_stream)
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    out = model(
                        gpu_inputs["input_ids"],
                        gpu_inputs["attention_mask"],
                        return_embedding=True,
                    )
            m1.record(compute_stream)
            model_events.append((m0, m1))
        else:
            gpu_inputs = {k: v.to(device) for k, v in cpu_inputs.items()}
            with torch.inference_mode():
                out = model(
                    gpu_inputs["input_ids"],
                    gpu_inputs["attention_mask"],
                    return_embedding=True,
                )

        if want == "decision":
            gpu_outputs.append(out["decision"])
        elif want == "embedding":
            gpu_outputs.append(out["embedding"])
        else:
            raise ValueError(want)
        cpu_submit_sum += time.perf_counter() - submit0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        live_cpu_inputs.clear()
        live_gpu_inputs.clear()

    launch_wall = time.perf_counter() - stage_start

    d2h0 = time.perf_counter()
    if gpu_outputs:
        merged = torch.cat(gpu_outputs, dim=0).detach().float().cpu()
    else:
        merged = torch.empty((0,))
    d2h_wall = time.perf_counter() - d2h0

    copy_ms = 0.0
    model_ms = 0.0
    if device.type == "cuda":
        for a, b in copy_events:
            copy_ms += float(a.elapsed_time(b))
        for a, b in model_events:
            model_ms += float(a.elapsed_time(b))

    return merged, token_counts, {
        "wall_s": launch_wall + d2h_wall,
        "launch_wall_s": launch_wall,
        "tokenizer_task_sum_s": tokenize_sum,
        "pin_memory_sum_s": pin_sum,
        "cpu_submit_sum_s": cpu_submit_sum,
        "h2d_cuda_ms": copy_ms,
        "model_cuda_ms": model_ms,
        "d2h_wall_s": d2h_wall,
        "batches": len(text_batches),
        "real_tokens": int(real_tokens_total),
        "padded_tokens": int(padded_tokens_total),
        "padding_efficiency": float(real_tokens_total / max(padded_tokens_total, 1)),
    }



def _semantic_fast_one_doc(engine, ps, valid):
    """Vectorized equivalent of selector semantic geometry for one document."""
    import torch

    if not valid:
        return {
            "paragraphs_embedded": 0,
            "coherence_observable": False,
            "adjacent_cosine_mean": None,
            "adjacent_cosine_min": None,
            "adjacent_boundary_max": None,
            "context_boundary_mean": 0.0,
            "context_boundary_max": 0.0,
            "tail_anomaly": 0.0,
            "best_boundary": None,
            "largest_coherent_region_ratio": 0.0,
            "second_coherent_region_ratio": 0.0,
            "substantial_region_count": 0,
            "substantial_region_coverage": 0.0,
            "semantic_regions": [],
            "region_threshold": 0.74,
            "region_size_basis": "token_mass",
            "total_embedded_chars": 0,
            "total_embedded_words": 0,
            "total_embedded_tokens": 0,
            "min_split_side_words": 100,
            "min_split_token_ratio": 0.20,
            "best_recoverable_split": None,
            "recoverable_split_score": 0.0,
            "adjacent_profile": [],
            "context_boundary_profile": [],
            "recoverable_split_profile": [],
        }

    ids = [int(i) for i, _ in valid]
    E = torch.stack([e.float() for _, e in valid], dim=0).contiguous()
    n = E.shape[0]
    norms = torch.linalg.vector_norm(E, dim=1).clamp_min(1e-12)
    En = E / norms[:, None]

    if n >= 2:
        sims_t = (En[:-1] * En[1:]).sum(dim=1)
        sims = [float(x) for x in sims_t.tolist()]
    else:
        sims_t = torch.empty(0, dtype=torch.float32)
        sims = []

    adjacent = [
        {
            "left_paragraph": ids[i],
            "right_paragraph": ids[i + 1],
            "cosine": sims[i],
            "boundary_score": 1.0 - sims[i],
        }
        for i in range(len(sims))
    ]

    prefix = torch.cat([torch.zeros((1, E.shape[1]), dtype=E.dtype), E.cumsum(dim=0)], dim=0)

    contextual = []
    if n >= 2:
        cuts = torch.arange(1, n, dtype=torch.long)
        l0 = torch.clamp(cuts - 3, min=0)
        l1 = cuts
        r0 = cuts
        r1 = torch.clamp(cuts + 3, max=n)
        left = (prefix[l1] - prefix[l0]) / (l1 - l0).to(E.dtype)[:, None]
        right = (prefix[r1] - prefix[r0]) / (r1 - r0).to(E.dtype)[:, None]
        den = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(right, dim=1)
        ctx = torch.where(den > 1e-12, (left * right).sum(dim=1) / den.clamp_min(1e-12), torch.zeros_like(den))
        for j, simv in enumerate(ctx.tolist(), start=1):
            contextual.append({
                "after_paragraph": ids[j - 1],
                "before_paragraph": ids[j],
                "context_cosine": float(simv),
                "context_boundary_score": 1.0 - float(simv),
            })

    min_adj = min(sims) if sims else None
    mean_adj = (sum(sims) / len(sims)) if sims else None
    boundary_scores = [x["context_boundary_score"] for x in contextual]
    best = max(contextual, key=lambda x: x["context_boundary_score"], default=None)

    tail_anomaly = 0.0
    if n >= 5:
        k = max(1, n // 5)
        body = E[:-k].mean(dim=0)
        tail = E[-k:].mean(dim=0)
        den = torch.linalg.vector_norm(body) * torch.linalg.vector_norm(tail)
        sim = float(torch.dot(body, tail) / den) if float(den) > 1e-12 else 0.0
        tail_anomaly = 1.0 - sim

    chars = []
    word_counts = []
    toks = []
    para_mass = {}
    for pid in ids:
        txt = ps[pid]
        c = len(txt)
        w = len(engine.words(txt))
        t = max(1, int(round(c / 3.5)))
        chars.append(c); word_counts.append(w); toks.append(t)
        para_mass[pid] = {"chars": c, "words": w, "tokens": t}

    def pref(xs):
        out = [0]
        z = 0
        for x in xs:
            z += int(x); out.append(z)
        return out

    pc, pw, pt = pref(chars), pref(word_counts), pref(toks)
    total_chars, total_words, total_tokens = pc[-1], pw[-1], pt[-1]

    def mass(a, b):
        c = pc[b]-pc[a]; w = pw[b]-pw[a]; t = pt[b]-pt[a]
        return {
            "paragraphs": b-a, "chars": c, "words": w, "tokens": t,
            "paragraph_ratio": (b-a)/max(n,1),
            "char_ratio": c/max(total_chars,1),
            "word_ratio": w/max(total_words,1),
            "token_ratio": t/max(total_tokens,1),
        }

    min_split_side_words = max(100, min(150, int(round(0.20 * total_words)))) if total_words else 100
    min_split_token_ratio = 0.20
    split_candidates = []
    if n >= 2:
        sim_prefix = [0.0]
        z=0.0
        for x in sims:
            z += float(x); sim_prefix.append(z)
        for cut in range(1, n):
            lm, rm = mass(0,cut), mass(cut,n)
            if (lm["token_ratio"] < min_split_token_ratio or rm["token_ratio"] < min_split_token_ratio or
                lm["words"] < min_split_side_words or rm["words"] < min_split_side_words):
                continue
            lcnt=max(cut-1,0); rcnt=max(n-cut-1,0)
            left_coh=(sim_prefix[cut-1]/lcnt) if lcnt else 0.0
            right_coh=((sim_prefix[n-1]-sim_prefix[cut])/rcnt) if rcnt else 0.0
            left_mean=(prefix[cut]-prefix[0])/float(cut)
            right_mean=(prefix[n]-prefix[cut])/float(n-cut)
            den=torch.linalg.vector_norm(left_mean)*torch.linalg.vector_norm(right_mean)
            cross=float(torch.dot(left_mean,right_mean)/den) if float(den)>1e-12 else 0.0
            token_balance=2.0*min(lm["token_ratio"],rm["token_ratio"])
            lq=max(0.0,min(1.0,(left_coh-0.74)/0.16)); rq=max(0.0,min(1.0,(right_coh-0.74)/0.16))
            sq=max(0.0,min(1.0,(0.78-cross)/0.30)); bq=max(0.0,min(1.0,token_balance))
            score=min(lq,rq)*sq*bq
            split_candidates.append({
                "after_paragraph":ids[cut-1],"before_paragraph":ids[cut],
                "left_paragraphs":cut,"right_paragraphs":n-cut,
                "left_chars":int(lm["chars"]),"right_chars":int(rm["chars"]),
                "left_words":int(lm["words"]),"right_words":int(rm["words"]),
                "left_tokens":int(lm["tokens"]),"right_tokens":int(rm["tokens"]),
                "left_paragraph_ratio":lm["paragraph_ratio"],"right_paragraph_ratio":rm["paragraph_ratio"],
                "left_char_ratio":lm["char_ratio"],"right_char_ratio":rm["char_ratio"],
                "left_word_ratio":lm["word_ratio"],"right_word_ratio":rm["word_ratio"],
                "left_token_ratio":lm["token_ratio"],"right_token_ratio":rm["token_ratio"],
                "min_required_side_words":min_split_side_words,"min_required_side_token_ratio":min_split_token_ratio,
                "left_coherence":left_coh,"right_coherence":right_coh,"cross_region_cosine":cross,
                "token_balance":token_balance,"left_quality":lq,"right_quality":rq,
                "separation_quality":sq,"recoverable_split_score":score,
            })
    best_split=max(split_candidates,key=lambda x:x["recoverable_split_score"],default=None)

    region_threshold=0.74
    ranges=[]
    start=0
    for edge,sim in enumerate(sims):
        if sim < region_threshold:
            ranges.append((start,edge+1)); start=edge+1
    ranges.append((start,n))
    regions=[]
    sim_prefix=[0.0]
    z=0.0
    for x in sims: z+=float(x); sim_prefix.append(z)
    for a,b in ranges:
        m=mass(a,b)
        cnt=max(b-a-1,0)
        coh=((sim_prefix[b-1]-sim_prefix[a])/cnt) if cnt else None
        regions.append({
            "start_paragraph":ids[a],"end_paragraph":ids[b-1],"paragraphs":int(m["paragraphs"]),
            "chars":int(m["chars"]),"words":int(m["words"]),"tokens":int(m["tokens"]),
            "paragraph_ratio":m["paragraph_ratio"],"char_ratio":m["char_ratio"],"word_ratio":m["word_ratio"],
            "token_ratio":m["token_ratio"],"ratio":m["token_ratio"],"coherence":coh,
        })
    rs=sorted(regions,key=lambda r:r["token_ratio"],reverse=True)
    largest=float(rs[0]["token_ratio"]) if rs else 0.0
    second=float(rs[1]["token_ratio"]) if len(rs)>1 else 0.0
    substantial=[r for r in regions if r["token_ratio"]>=0.20 and r["words"]>=min_split_side_words and (r["coherence"] is None or float(r["coherence"])>=0.74)]

    return {
        "paragraphs_embedded":n,"coherence_observable":len(sims)>0,
        "adjacent_cosine_mean":mean_adj,"adjacent_cosine_min":min_adj,
        "adjacent_boundary_max":(1.0-min_adj) if min_adj is not None else None,
        "context_boundary_mean":sum(boundary_scores)/len(boundary_scores) if boundary_scores else 0.0,
        "context_boundary_max":max(boundary_scores,default=0.0),"tail_anomaly":tail_anomaly,
        "best_boundary":best,"largest_coherent_region_ratio":largest,"second_coherent_region_ratio":second,
        "substantial_region_count":len(substantial),"substantial_region_coverage":sum(float(r["token_ratio"]) for r in substantial),
        "semantic_regions":regions,"region_threshold":region_threshold,"region_size_basis":"token_mass",
        "total_embedded_chars":total_chars,"total_embedded_words":total_words,"total_embedded_tokens":total_tokens,
        "min_split_side_words":min_split_side_words,"min_split_token_ratio":min_split_token_ratio,
        "best_recoverable_split":best_split,"recoverable_split_score":float(best_split["recoverable_split_score"]) if best_split else 0.0,
        "adjacent_profile":adjacent,"context_boundary_profile":contextual,"recoverable_split_profile":split_candidates,
    }

def oflow_attach_semantic(
    sample: list[dict[str, Any]],
    args,
    engine,
    *,
    tokenizer_workers: int,
    prefetch_depth: int,
    pin_memory: bool,
    semantic_workers: int = 1,
    gpu_mode: str = "native",
    bucket_window: int = 256,
):
    import torch

    total0 = time.perf_counter()
    (
        checkpoint_path,
        tokenizer_source,
        device,
        model,
        tokenizer,
        schema,
        train_args,
    ) = _resolve_runtime(engine, args)

    use_bucket = gpu_mode in {"bucket", "bucket_compile"}
    model = _maybe_compile_model(model, gpu_mode == "bucket_compile")

    use_amp = device.type == "cuda" and (args.bf16 or args.fp16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # -------------------------------------------------------------------------
    # Document pass: CPU tokenizer prefetch + H2D copy stream + no per-batch D2H
    # -------------------------------------------------------------------------
    if use_bucket:
        _doc_items = _bounded_length_order(
            [(i, x["text"]) for i, x in enumerate(sample)],
            lambda z: z[1],
            bucket_window,
        )
        doc_text_batches=[]
        doc_index_batches=[]
        for _start in range(0, len(_doc_items), args.document_batch_size):
            _chunk=_doc_items[_start:_start+args.document_batch_size]
            doc_text_batches.append([z[1] for z in _chunk])
            doc_index_batches.append([z[0] for z in _chunk])
    else:
        doc_text_batches, _ = _make_cpu_batches(
            sample,
            args.document_batch_size,
            lambda x: x["text"],
        )
        doc_index_batches=[]
        for _start in range(0, len(sample), args.document_batch_size):
            doc_index_batches.append(list(range(_start, min(_start+args.document_batch_size, len(sample)))))

    doc_logits, doc_token_counts_nested, doc_stats = _gpu_launch_batches(
        model=model,
        tokenizer=tokenizer,
        text_batches=doc_text_batches,
        max_length=args.document_max_length,
        tokenizer_workers=tokenizer_workers,
        prefetch_depth=prefetch_depth,
        pin_memory=pin_memory,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        want="decision",
    )

    # Restore original document order after optional length bucketing.
    _flat_doc_idx = [i for b in doc_index_batches for i in b]
    _flat_tok = [x for batch in doc_token_counts_nested for x in batch]
    if _flat_doc_idx != list(range(len(sample))):
        _restored_logits = torch.empty_like(doc_logits)
        _restored_counts = [0] * len(sample)
        for _pos, _orig in enumerate(_flat_doc_idx):
            _restored_logits[_orig] = doc_logits[_pos]
            _restored_counts[_orig] = _flat_tok[_pos]
        doc_logits = _restored_logits
        flat_tok_counts = _restored_counts
    else:
        flat_tok_counts = _flat_tok

    probs = torch.softmax(doc_logits, dim=-1)
    probs_list = probs.tolist()

    for item, row_probs, tok_count in zip(sample, probs_list, flat_tok_counts):
        order = sorted(range(len(row_probs)), key=lambda i: row_probs[i], reverse=True)
        item["metadata"]["tokenization"] = {
            "mmbert_document_tokens_visible": int(tok_count),
            "mmbert_document_max_length": args.document_max_length,
            "mmbert_document_truncated": int(tok_count) >= args.document_max_length,
        }
        item["metadata"]["old_mmbert"] = {
            "decision_probs": {
                label: float(row_probs[i]) for i, label in enumerate(schema.decisions)
            },
            "predicted": schema.decisions[order[0]],
            "confidence": float(row_probs[order[0]]),
            "second": schema.decisions[order[1]] if len(order) > 1 else None,
            "second_probability": float(row_probs[order[1]]) if len(order) > 1 else None,
            "margin": float(row_probs[order[0]] - row_probs[order[1]]) if len(order) > 1 else 1.0,
        }

    # -------------------------------------------------------------------------
    # Build paragraph work on CPU
    # -------------------------------------------------------------------------
    para_build0 = time.perf_counter()
    flat: list[tuple[int, int, str]] = []
    doc_paragraphs: dict[int, list[str]] = {}
    for doc_idx, item in enumerate(sample):
        ps = engine.paragraphs(item["text"])
        if len(ps) > args.max_paragraphs_per_doc:
            n = args.max_paragraphs_per_doc
            chosen_idx = sorted(
                set(round(i * (len(ps) - 1) / max(n - 1, 1)) for i in range(n))
            )
            ps = [ps[i] for i in chosen_idx]
        doc_paragraphs[doc_idx] = ps
        for para_idx, p in enumerate(ps):
            if len(engine.words(p)) >= 3:
                flat.append((doc_idx, para_idx, p))
    para_build_s = time.perf_counter() - para_build0

    if use_bucket:
        flat = _bounded_length_order(flat, lambda z: z[2], bucket_window)

    para_batches: list[list[str]] = []
    para_key_batches: list[list[tuple[int, int]]] = []
    for start in range(0, len(flat), args.paragraph_batch_size):
        chunk = flat[start:start + args.paragraph_batch_size]
        para_batches.append([x[2] for x in chunk])
        para_key_batches.append([(x[0], x[1]) for x in chunk])

    para_embs, _, para_stats = _gpu_launch_batches(
        model=model,
        tokenizer=tokenizer,
        text_batches=para_batches,
        max_length=args.paragraph_max_length,
        tokenizer_workers=tokenizer_workers,
        prefetch_depth=prefetch_depth,
        pin_memory=pin_memory,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        want="embedding",
    )

    flat_keys = [k for batch in para_key_batches for k in batch]
    embeddings: dict[tuple[int, int], Any] = {
        key: emb for key, emb in zip(flat_keys, para_embs)
    }

    # -------------------------------------------------------------------------
    # Vectorized semantic postprocess. Same output schema, far fewer tiny
    # torch.stack/cosine calls. Optional document-level threading is autotuned.
    # -------------------------------------------------------------------------
    semantic0 = time.perf_counter()

    jobs = []
    for doc_idx, item in enumerate(sample):
        ps = doc_paragraphs[doc_idx]
        valid = [(i, embeddings[(doc_idx, i)]) for i in range(len(ps)) if (doc_idx, i) in embeddings]
        jobs.append((doc_idx, ps, valid))

    def _do(job):
        doc_idx, ps, valid = job
        return doc_idx, _semantic_fast_one_doc(engine, ps, valid)

    if semantic_workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=semantic_workers, thread_name_prefix="oflow-sem") as ex:
            for doc_idx, sem in ex.map(_do, jobs):
                sample[doc_idx]["metadata"]["semantic_structure"] = sem
    else:
        for job in jobs:
            doc_idx, sem = _do(job)
            sample[doc_idx]["metadata"]["semantic_structure"] = sem

    semantic_cpu_s = time.perf_counter() - semantic0
    total_s = time.perf_counter() - total0

    return {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "tokenizer": str(tokenizer_source),
        "model_name": train_args.get("model", train_args.get("model_name", None)),
        "device": str(device),
        "decisions": schema.decisions,
        "oflow_stats": {
            "total_s": total_s,
            "document_pass": doc_stats,
            "paragraph_build_s": para_build_s,
            "paragraph_pass": para_stats,
            "semantic_cpu_s": semantic_cpu_s,
            "paragraphs_total": len(flat),
            "tokenizer_workers": tokenizer_workers,
            "prefetch_depth": prefetch_depth,
            "pin_memory": bool(pin_memory),
            "semantic_workers": int(semantic_workers),
            "gpu_mode": gpu_mode,
            "bucket_window": int(bucket_window),
        },
    }


# =============================================================================
# Data/sample prep and equivalence
# =============================================================================


def prepare_sample(engine, input_path: Path, args):
    # Attributes expected by selector helpers.
    args.text_key = ""
    args.sample_size = args.sample_docs

    sample, scan = engine.reservoir_sample(input_path, args)
    t0 = time.perf_counter()
    for item in sample:
        if hasattr(engine, "base_document_metadata_fast"):
            surface = engine.base_document_metadata_fast(item["text"], args.ngram_n)
        else:
            surface = engine.base_document_metadata(
                item["text"], args.ngram_n, args.near_dup_threshold
            )
        item["metadata"] = {"surface": surface}
    surface_s = time.perf_counter() - t0
    return sample, scan, surface_s


def _compare(a, b, state, path="root"):
    import numbers

    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        da = float(a)
        db = float(b)
        if math.isfinite(da) and math.isfinite(db):
            diff = abs(da - db)
            state["max_abs_float_diff"] = max(state["max_abs_float_diff"], diff)
            if diff > state["float_tolerance"]:
                state["value_mismatches"] += 1
                if len(state["examples"]) < 8:
                    state["examples"].append(f"{path}: {da} vs {db} diff={diff}")
        elif da != db:
            state["value_mismatches"] += 1
        return

    if type(a) is not type(b):
        state["structure_mismatches"] += 1
        if len(state["examples"]) < 8:
            state["examples"].append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        return

    if isinstance(a, dict):
        ka, kb = set(a), set(b)
        if ka != kb:
            state["structure_mismatches"] += len(ka ^ kb)
            if len(state["examples"]) < 8:
                state["examples"].append(f"{path}: key delta={sorted(ka ^ kb)[:10]}")
        for k in ka & kb:
            _compare(a[k], b[k], state, f"{path}.{k}")
        return

    if isinstance(a, list):
        if len(a) != len(b):
            state["structure_mismatches"] += 1
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, state, f"{path}[{i}]")
        return

    if a != b:
        state["value_mismatches"] += 1
        if len(state["examples"]) < 8:
            state["examples"].append(f"{path}: {a!r} vs {b!r}")


def compare_semantic(reference, candidate, tol=5e-4):
    state = {
        "float_tolerance": tol,
        "max_abs_float_diff": 0.0,
        "structure_mismatches": 0,
        "value_mismatches": 0,
        "examples": [],
    }
    if len(reference) != len(candidate):
        state["structure_mismatches"] += 1
        return state

    for i, (a, b) in enumerate(zip(reference, candidate)):
        aa = {
            "old_mmbert": a.get("metadata", {}).get("old_mmbert"),
            "semantic_structure": a.get("metadata", {}).get("semantic_structure"),
        }
        bb = {
            "old_mmbert": b.get("metadata", {}).get("old_mmbert"),
            "semantic_structure": b.get("metadata", {}).get("semantic_structure"),
        }
        _compare(aa, bb, state, f"doc[{i}]")

    state["equivalent"] = (
        state["structure_mismatches"] == 0 and state["value_mismatches"] == 0
    )
    return state


# =============================================================================
# Benchmark helpers
# =============================================================================


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def approx_sample_tokens(sample) -> int:
    return int(sum(len(x["text"]) for x in sample) / 3.5)


def run_reference(engine, sample, args, doc_bs: int, para_bs: int):
    args.document_batch_size = doc_bs
    args.paragraph_batch_size = para_bs
    work = copy.deepcopy(sample)
    cuda_sync()
    t0 = time.perf_counter()
    engine.attach_mmbert_metadata(work, args)
    cuda_sync()
    dt = time.perf_counter() - t0
    return work, dt


def run_oflow(engine, sample, args, c: Candidate, pin_memory: bool):
    args.document_batch_size = c.document_batch
    args.paragraph_batch_size = c.paragraph_batch
    work = copy.deepcopy(sample)
    cuda_sync()
    t0 = time.perf_counter()
    info = oflow_attach_semantic(
        work,
        args,
        engine,
        tokenizer_workers=c.tokenizer_workers,
        prefetch_depth=c.prefetch_depth,
        pin_memory=pin_memory,
        semantic_workers=c.semantic_workers,
        gpu_mode=c.gpu_mode,
        bucket_window=args.bucket_window,
    )
    cuda_sync()
    dt = time.perf_counter() - t0
    return work, dt, info["oflow_stats"]


def diagnose(best: dict[str, Any], system: dict[str, Any]) -> list[str]:
    notes = []
    stats = best.get("stats", {})
    total = max(float(stats.get("total_s", best.get("seconds", 0.0))), 1e-9)
    doc = stats.get("document_pass", {})
    para = stats.get("paragraph_pass", {})
    semantic = float(stats.get("semantic_cpu_s", 0.0))
    para_build = float(stats.get("paragraph_build_s", 0.0))

    model_cuda = (
        float(doc.get("model_cuda_ms", 0.0))
        + float(para.get("model_cuda_ms", 0.0))
    ) / 1000.0
    h2d_cuda = (
        float(doc.get("h2d_cuda_ms", 0.0))
        + float(para.get("h2d_cuda_ms", 0.0))
    ) / 1000.0
    tokenizer_sum = (
        float(doc.get("tokenizer_task_sum_s", 0.0))
        + float(para.get("tokenizer_task_sum_s", 0.0))
    )

    if model_cuda / total < 0.55:
        notes.append(
            "CUDA model compute is below ~55% of measured semantic wall time; there is room outside model.forward()."
        )
    else:
        notes.append(
            "Model CUDA time dominates this run; large runtime-only gains may be harder from this configuration."
        )

    if tokenizer_sum > model_cuda * 0.5:
        notes.append(
            "Tokenizer work is material relative to model compute; CPU prefetch/worker tuning is relevant."
        )
    if semantic / total > 0.20:
        notes.append(
            "CPU semantic postprocess is material; vectorizing cosine/region work is a likely next target."
        )
    if para_build / total > 0.10:
        notes.append(
            "Paragraph extraction/work construction is visible; move it to a producer stage or parallel worker pool."
        )
    if h2d_cuda / total > 0.10:
        notes.append(
            "Host-to-device transfer is visible; pinned buffers/copy overlap are worth keeping."
        )

    notes.append(
        "v0 does not yet overlap semantic CPU postprocess of batch N with GPU inference of batch N+1; that is a deliberate next-step opportunity."
    )
    return notes



# =============================================================================
# Optional Textual welcome screen
# =============================================================================

_OFLOW_WORDMARK = r"""
 ██████╗ ███████╗██╗      ██████╗ ██╗    ██╗
██╔═══██╗██╔════╝██║     ██╔═══██╗██║    ██║
██║   ██║█████╗  ██║     ██║   ██║██║ █╗ ██║
██║   ██║██╔══╝  ██║     ██║   ██║██║███╗██║
╚██████╔╝██║     ███████╗╚██████╔╝╚███╔███╔╝
 ╚═════╝ ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
""".strip("\n")


def _launch_welcome_if_needed() -> None:
    """No args = interactive launcher. Any CLI args keep classic script behaviour."""
    if len(sys.argv) > 1:
        return

    try:
        from textual.app import App, ComposeResult
        from textual.containers import Vertical, Horizontal
        from textual.widgets import Static, Button, Footer
        from textual.reactive import reactive
        import math as _math

        class AmbientBand(Static):
            phase = reactive(0.0)
            def on_mount(self):
                self.set_interval(0.07, self._advance)
                self._paint()
            def _advance(self):
                self.phase += 0.23
                self._paint()
            def _paint(self):
                # Content animation only: no layout movement.
                blocks=[]
                for i in range(28):
                    x=(1.0+_math.sin(self.phase-i*0.42))*0.5
                    blocks.append("█" if x>0.78 else "▓" if x>0.52 else "▒" if x>0.28 else "░")
                self.update("".join(blocks))

        class Welcome(App):
            CSS = """
            Screen { align: center middle; background: #090b10; }
            #card { width: 72; height: auto; padding: 1 3; border: round #5f8cff; background: #10141d; }
            #mark { color: #8fb2ff; text-style: bold; content-align: center middle; }
            #compact { color: #8fb2ff; text-style: bold; content-align: center middle; display: none; }
            #band { color: #6f98ff; content-align: center middle; height: 1; margin: 1 0; }
            #tag { color: #aab4c8; content-align: center middle; margin-bottom: 1; }
            #quick { width: 100%; height: 3; margin-top: 1; }
            #hint { color: #6f788c; content-align: center middle; margin-top: 1; }
            """
            selected = None
            BINDINGS=[("q","quit","Quit")]
            def compose(self) -> ComposeResult:
                with Vertical(id="card"):
                    yield Static(_OFLOW_WORDMARK, id="mark")
                    yield Static("◆ OFlow — adaptive pipeline runtime", id="compact")
                    yield AmbientBand(id="band")
                    yield Static("Find the bottleneck. Fill the bubbles. Preserve the output.", id="tag")
                    yield Button("QUICK  ·  probe + reference + autotune", id="quick", variant="primary")
                    yield Static("Quick is the first smoke path. More modes come later.", id="hint")
                yield Footer()
            def on_button_pressed(self, event: Button.Pressed):
                if event.button.id == "quick":
                    self.selected="quick"
                    self.exit()
            def on_resize(self, event):
                narrow = event.size.width < 60
                self.query_one("#mark").display = not narrow
                self.query_one("#compact").display = narrow

        app=Welcome()
        app.run()
        if app.selected == "quick":
            sys.argv.extend(["--bf16", "--full-probe"])
        else:
            raise SystemExit(0)
    except ImportError:
        print("OFlow QUICK launcher (install 'textual' for the clickable TUI)\n")
        print("  [1] QUICK  - probe + reference + autotune")
        print("  [Q] Quit")
        choice=input("\nSelect: ").strip().lower()
        if choice in {"1","quick",""}:
            sys.argv.extend(["--bf16", "--full-probe"])
        else:
            raise SystemExit(0)

# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 / --fp16")

    here = Path(__file__).resolve().parent
    input_path = resolve_path(args.input, here)
    engine_path = resolve_path(args.engine_script, here)
    pipeline_path = resolve_path(args.pipeline_script, here)
    checkpoint = resolve_path(args.checkpoint, here)
    tokenizer = Path(args.tokenizer)
    if args.tokenizer and not tokenizer.exists():
        candidate = here / tokenizer
        if candidate.exists():
            tokenizer = candidate.resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)
    if not pipeline_path.exists():
        raise FileNotFoundError(pipeline_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    args.input = str(input_path)
    args.checkpoint = str(checkpoint)
    args.tokenizer = str(tokenizer) if tokenizer.exists() else args.tokenizer

    print("=" * 108)
    print("OFlow v3-gpu — GPU SHAPE / KERNEL DIAGNOSTIC")
    print("=" * 108)

    system = basic_system_probe()
    print("[system]")
    for k, v in system.items():
        print(f"  {k:24s}: {v}")

    probes = {}
    if args.full_probe:
        print("\n[full-probe] storage...")
        probes["storage"] = storage_probe(input_path, args.probe_mb, args.gzip_probe_records)
        for k, v in probes["storage"].items():
            print(f"  {k:24s}: {v}")

        print("[full-probe] H2D / D2H...")
        probes["transfer"] = full_transfer_probe(args.probe_mb)
        for k, v in probes["transfer"].items():
            print(f"  {k:24s}: {v}")

    engine = import_file("oflow_selector", engine_path)
    pipeline = import_file("oflow_pipeline", pipeline_path)

    # Install the requested backend while keeping the selector API unchanged.
    if args.backend == "oris":
        if not hasattr(pipeline, "make_oris_filter_loader"):
            raise RuntimeError(
                "Pipeline script has no make_oris_filter_loader(); point --pipeline-script at the ORIS production pipeline."
            )
        engine.load_mmbert = pipeline.make_oris_filter_loader()
    else:
        # mmBERT uses the selector's native loader.
        pass

    patch = pipeline.PersistentTorchPatch(engine, checkpoint)

    try:
        # Selector helper attributes used by reservoir/sample paths.
        args.max_scan = int(args.max_scan)
        args.sample_size = int(args.sample_docs)
        args.text_key = ""

        print("\n[sample] reservoir + surface metadata...")
        sample, scan, surface_s = prepare_sample(engine, input_path, args)
        approx_tok = approx_sample_tokens(sample)
        print(
            f"  docs={len(sample):,} approx_tokens={approx_tok:,} "
            f"surface={surface_s:.3f}s ({len(sample)/max(surface_s,1e-9):.1f} docs/s)"
        )
        print(f"  scan={scan}")

        # Warm model/tokenizer/cache once.
        warm = copy.deepcopy(sample[: max(1, min(args.warmup_docs, len(sample)))])
        args.document_batch_size = min(8, len(warm))
        args.paragraph_batch_size = 64
        print("\n[warmup] model/tokenizer/CUDA kernels...")
        cuda_sync()
        t0 = time.perf_counter()
        engine.attach_mmbert_metadata(warm, args)
        cuda_sync()
        print(f"  {time.perf_counter()-t0:.3f}s")

        bdoc, bpara = parse_pair(args.baseline_config)
        print("\n[reference] current sequential selector path...")
        reference_runs = []
        reference_work = None
        for r in range(args.repeats):
            work, dt = run_reference(engine, sample, args, bdoc, bpara)
            reference_work = work
            row = {
                "repeat": r + 1,
                "seconds": dt,
                "docs_per_second": len(sample) / max(dt, 1e-9),
                "approx_tokens_per_second": approx_tok / max(dt, 1e-9),
                "document_batch": bdoc,
                "paragraph_batch": bpara,
            }
            reference_runs.append(row)
            print(
                f"  repeat={r+1} {row['docs_per_second']:.2f} docs/s "
                f"~{row['approx_tokens_per_second']:,.0f} tok/s ({dt:.3f}s)"
            )

        reference_dps = statistics.median(x["docs_per_second"] for x in reference_runs)

        results = []
        _base_candidates = parse_candidates(args.candidates)
        _modes = [x.strip() for x in args.gpu_modes.split(",") if x.strip()]
        _valid_modes = {"native", "bucket", "bucket_compile"}
        if any(x not in _valid_modes for x in _modes):
            raise ValueError(f"Bad --gpu-modes; choose from {sorted(_valid_modes)}")
        candidates=[]
        # Keep the search bounded: GPU modes are crossed with the two most useful base shapes.
        # The user can still override --candidates explicitly.
        for _c in _base_candidates:
            for _mode in _modes:
                candidates.append(Candidate(
                    _c.document_batch, _c.paragraph_batch, _c.tokenizer_workers,
                    _c.prefetch_depth, _c.semantic_workers, _mode
                ))
        print("\n[autotune] OFlow GPU candidates (shape bucketing / compile)...")
        for idx, c in enumerate(candidates, 1):
            try:
                run_rows = []
                last_work = None
                last_stats = None
                for r in range(args.repeats):
                    work, dt, stats = run_oflow(engine, sample, args, c, args.pin_memory)
                    last_work = work
                    last_stats = stats
                    dps = len(sample) / max(dt, 1e-9)
                    run_rows.append({
                        "repeat": r + 1,
                        "seconds": dt,
                        "docs_per_second": dps,
                        "approx_tokens_per_second": approx_tok / max(dt, 1e-9),
                    })

                dps = statistics.median(x["docs_per_second"] for x in run_rows)
                tps = statistics.median(x["approx_tokens_per_second"] for x in run_rows)
                eq = compare_semantic(reference_work, last_work)
                result = {
                    "candidate": asdict(c),
                    "label": c.label,
                    "seconds": statistics.median(x["seconds"] for x in run_rows),
                    "docs_per_second": dps,
                    "approx_tokens_per_second": tps,
                    "speedup_vs_reference": dps / max(reference_dps, 1e-9),
                    "equivalence": eq,
                    "stats": last_stats,
                    "runs": run_rows,
                }
                results.append(result)

                print(
                    f"  [{idx:02d}] {c.label:56s} -> "
                    f"{dps:7.2f} docs/s  {result['speedup_vs_reference']:5.2f}x  "
                    f"equiv={'YES' if eq.get('equivalent') else 'NO'}  "
                    f"maxΔ={eq['max_abs_float_diff']:.2e}"
                )
                if not eq.get("equivalent") and eq.get("examples"):
                    print("       equivalence examples:")
                    for ex in eq["examples"][:3]:
                        print("       ", ex)

            except Exception as exc:
                print(f"  [{idx:02d}] {c.label} -> FAILED: {exc!r}")
                results.append({
                    "candidate": asdict(c),
                    "label": c.label,
                    "failed": True,
                    "error": repr(exc),
                })

        valid = [
            x for x in results
            if not x.get("failed") and x.get("equivalence", {}).get("equivalent")
        ]
        if not valid:
            valid = [x for x in results if not x.get("failed")]
        if not valid:
            raise RuntimeError("All OFlow candidates failed")

        best = max(valid, key=lambda x: x["docs_per_second"])
        diagnosis = diagnose(best, system)

        print("\n" + "=" * 108)
        print("OFLOW v2-long RESULT")
        print("=" * 108)
        print(f"Reference : {reference_dps:.2f} docs/s")
        print(
            f"Best      : {best['docs_per_second']:.2f} docs/s | "
            f"{best['speedup_vs_reference']:.3f}x | {best['label']}"
        )
        print(f"Equivalent: {best.get('equivalence', {}).get('equivalent')}")

        st = best.get("stats", {})
        if st:
            doc = st.get("document_pass", {})
            para = st.get("paragraph_pass", {})
            print("\n[best stage telemetry]")
            print(f"  document pass wall : {doc.get('wall_s', 0.0):.3f}s")
            print(f"  document model CUDA: {doc.get('model_cuda_ms', 0.0):.1f}ms | pad-eff={doc.get('padding_efficiency', 0.0):.1%}")
            print(f"  document tokenizer : {doc.get('tokenizer_task_sum_s', 0.0):.3f}s task-sum")
            print(f"  paragraph build    : {st.get('paragraph_build_s', 0.0):.3f}s")
            print(f"  paragraph pass wall: {para.get('wall_s', 0.0):.3f}s")
            print(f"  paragraph model CUDA: {para.get('model_cuda_ms', 0.0):.1f}ms | pad-eff={para.get('padding_efficiency', 0.0):.1%}")
            print(f"  paragraph tokenizer: {para.get('tokenizer_task_sum_s', 0.0):.3f}s task-sum")
            print(f"  semantic CPU       : {st.get('semantic_cpu_s', 0.0):.3f}s")
            print(f"  paragraphs         : {st.get('paragraphs_total', 0):,}")

        print("\n[diagnosis]")
        for note in diagnosis:
            print("  -", note)

        report = {
            "oflow_version": "v2-long",
            "backend": args.backend,
            "input": str(input_path),
            "checkpoint": str(checkpoint),
            "system": system,
            "full_probe": probes,
            "sample": {
                "documents": len(sample),
                "approx_tokens": approx_tok,
                "scan": scan,
                "surface_seconds": surface_s,
            },
            "reference": {
                "document_batch": bdoc,
                "paragraph_batch": bpara,
                "median_docs_per_second": reference_dps,
                "runs": reference_runs,
            },
            "candidates": results,
            "best": best,
            "diagnosis": diagnosis,
        }

        output = Path(args.output)
        if not output.is_absolute():
            output = Path.cwd() / output
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {output.resolve()}")
        print("=" * 108)

    finally:
        patch.close()



# =============================================================================
# OFlow v4 MAX — exact token-budget batching + semantic/GPU overlap + SDPA tune
# =============================================================================

from contextlib import nullcontext

# =============================================================================
# OFlow-Kania v2
# =============================================================================
# This keeps all v4 public symbols/API intact so `oflow_bakeoff.py` and its v5
# scheduler can import it unchanged. The v2 change is deliberately narrow:
# paragraph embedding batches use 8192-item tokenizer lookahead plus a dynamic
# >=88% real/padded-token efficiency floor (after a minimum of 16 rows).
# Document batching is unchanged.
# =============================================================================


@dataclass
class MaxCandidate:
    doc_token_budget: int = 16_384
    doc_max_batch: int = 32
    para_token_budget: int = 24_576
    para_max_batch: int = 256
    tokenizer_workers: int = 1
    prefetch_depth: int = 2
    tokenize_window: int = 2048
    semantic_chunk_docs: int = 256
    semantic_workers: int = 1
    pad_multiple: int = 8
    doc_sdpa: str = "auto"
    para_sdpa: str = "auto"
    fast_adapter: bool = True

    @property
    def label(self) -> str:
        return (
            f"docTB={self.doc_token_budget} docMax={self.doc_max_batch} "
            f"paraTB={self.para_token_budget} paraMax={self.para_max_batch} "
            f"win={self.tokenize_window} chunk={self.semantic_chunk_docs} "
            f"pad={self.pad_multiple} sdpa={self.doc_sdpa}/{self.para_sdpa}"
        )


def parse_args_v4():
    p = argparse.ArgumentParser(
        "OFlow Kania v2 — padding-aware paragraph batching / overlap / SDPA autotune"
    )
    p.add_argument("--input", default=r"data\madlad_stage0\raw\pl_clean_0000.jsonl.gz")
    p.add_argument("--engine-script", default="v2_candidate_selector.py")
    p.add_argument("--pipeline-script", default="v2_stage_v2_pipeline_oris.py")
    p.add_argument("--backend", choices=["oris", "mmbert"], default="oris")
    p.add_argument(
        "--checkpoint",
        default=r"runs\oris_vs_mmbert_v4_1024\oris_small_c\best_model.pt",
    )
    p.add_argument(
        "--tokenizer",
        default=r"runs\oris_vs_mmbert_v4_1024\oris_small_c\tokenizer",
    )

    p.add_argument("--sample-docs", type=int, default=2048)
    p.add_argument("--max-scan", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--min-chars", type=int, default=300)
    p.add_argument("--max-chars", type=int, default=100_000)
    p.add_argument("--ngram-n", type=int, default=8)
    p.add_argument("--near-dup-threshold", type=float, default=0.86)
    p.add_argument("--document-max-length", type=int, default=1024)
    p.add_argument("--paragraph-max-length", type=int, default=256)
    p.add_argument("--max-paragraphs-per-doc", type=int, default=96)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--baseline-config", default="8x64")
    p.add_argument("--warmup-docs", type=int, default=64)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")

    # Phase-1 search space. Kept deliberately compact; later phases tune kernels
    # and overlap around the winning token-budget shape.
    p.add_argument(
        "--shape-presets",
        default=(
            "12288x24x12288x192,"
            "16384x32x16384x256,"
            "16384x32x24576x256,"
            "16384x32x32768x256,"
            "24576x32x24576x256"
        ),
        help="docTokenBudget x docMaxBatch x paraTokenBudget x paraMaxBatch",
    )
    p.add_argument("--tokenizer-workers", type=int, default=1)
    p.add_argument("--prefetch-depth", type=int, default=2)
    p.add_argument("--tokenize-window", type=int, default=2048)
    p.add_argument("--semantic-chunk-docs", type=int, default=256)
    p.add_argument("--semantic-workers", type=int, default=1)
    p.add_argument("--pad-multiple", type=int, default=8)

    p.add_argument(
        "--kernel-backends",
        default="auto,cudnn,flash,efficient",
        help="SDPA backends tested after shape search",
    )
    p.add_argument(
        "--chunk-search",
        default="128,256,512",
        help="semantic/GPU overlap chunk sizes tested around the winner",
    )
    p.add_argument(
        "--pad-search",
        default="8,16,32",
        help="sequence padding multiples tested around the winner",
    )

    p.add_argument("--legacy-anchor", action="store_true", default=True)
    p.add_argument("--no-legacy-anchor", dest="legacy_anchor", action="store_false")
    p.add_argument("--full-probe", action="store_true", default=False)
    p.add_argument("--probe-mb", type=int, default=128)
    p.add_argument("--gzip-probe-records", type=int, default=5000)
    p.add_argument("--output", default="oflow_v4_max_report.json")
    return p.parse_args()


def _v4_parse_shape_presets(text: str) -> list[tuple[int, int, int, int]]:
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        xs = [int(x) for x in raw.lower().split("x")]
        if len(xs) != 4 or min(xs) <= 0:
            raise ValueError(
                f"Bad shape preset {raw!r}; expected docTBxdocMaxxparaTBxparaMax"
            )
        out.append(tuple(xs))
    if not out:
        raise ValueError("No --shape-presets")
    return out


def _v4_round_up(x: int, multiple: int, cap: int) -> int:
    multiple = max(1, int(multiple))
    y = ((int(x) + multiple - 1) // multiple) * multiple
    return min(max(1, y), int(cap))


def _v4_tokenize_window(
    tokenizer,
    items: list[tuple[Any, str]],
    max_length: int,
    token_budget: int,
    max_batch: int,
    pad_multiple: int,
    min_padding_efficiency: float = 0.0,
    efficiency_min_batch: int = 16,
):
    """Tokenize one lookahead window and build exact-length token-budget batches.

    Kania v2 addition
    -----------------
    The original v4 greedily filled a batch up to max_batch/token_budget after
    sorting by descending length. That is excellent for document sequences, but
    for tens of thousands of short paragraphs it can place a long-ish paragraph
    together with many much shorter rows, making the GPU spend a large fraction
    of time on padding.

    When ``min_padding_efficiency > 0`` this packer closes a batch early once
    adding the next (shorter) sequence would push real/padded token efficiency
    below the requested floor. A small minimum batch size prevents pathological
    tiny batches. The operation is semantics-preserving: examples, tokenization,
    and model math do not change; only batch composition changes.
    """
    import torch

    t0 = time.perf_counter()
    texts = [x[1] for x in items]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
        verbose=False,
    )
    token_s = time.perf_counter() - t0

    records = []
    for (key, _text), ids, mask in zip(
        items,
        enc["input_ids"],
        enc.get("attention_mask", [[1] * len(x) for x in enc["input_ids"]]),
    ):
        ids = [int(x) for x in ids]
        mask = [int(x) for x in mask]
        if not ids:
            continue
        real_tokens = int(sum(mask))
        records.append((key, ids, mask, len(ids), real_tokens))

    # Exact descending length is still the base ordering.
    records.sort(key=lambda z: z[3], reverse=True)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.sep_token_id
    if pad_id is None:
        pad_id = 0
    pad_id = int(pad_id)

    min_eff = float(min_padding_efficiency)
    min_eff = max(0.0, min(1.0, min_eff))
    eff_min_batch = max(1, int(efficiency_min_batch))

    batches = []
    pack0 = time.perf_counter()
    i = 0

    while i < len(records):
        first_len = records[i][3]
        padded_len = _v4_round_up(first_len, pad_multiple, max_length)
        hard_capacity = max(1, int(token_budget) // max(padded_len, 1))
        hard_capacity = max(1, min(int(max_batch), hard_capacity))

        # Padding-aware greedy fill.
        j = i
        real_sum = 0
        chosen = []

        while j < len(records) and len(chosen) < hard_capacity:
            row = records[j]
            proposed_n = len(chosen) + 1
            proposed_real = real_sum + int(row[4])
            proposed_padded = proposed_n * padded_len
            proposed_eff = proposed_real / max(proposed_padded, 1)

            # Always allow enough rows to keep kernels reasonably large.
            # After that, stop before a much shorter tail drags efficiency down.
            if (
                chosen
                and min_eff > 0.0
                and len(chosen) >= eff_min_batch
                and proposed_eff < min_eff
            ):
                break

            chosen.append(row)
            real_sum = proposed_real
            j += 1

        # Defensive fallback: always consume at least one row.
        if not chosen:
            chosen = [records[i]]
            j = i + 1

        i = j

        actual_max = max(z[3] for z in chosen)
        padded_len = _v4_round_up(actual_max, pad_multiple, max_length)

        id_rows = []
        mask_rows = []
        real = 0
        for key, ids, mask, ln, _real_tokens in chosen:
            ids = ids[:padded_len]
            mask = mask[:padded_len]
            n = len(ids)
            real += int(sum(mask))
            if n < padded_len:
                pad_n = padded_len - n
                ids = ids + [pad_id] * pad_n
                mask = mask + [0] * pad_n
            id_rows.append(ids)
            mask_rows.append(mask)

        cpu_inputs = {
            "input_ids": torch.tensor(id_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        }
        batches.append({
            "keys": [z[0] for z in chosen],
            "item_tokens": [int(z[4]) for z in chosen],
            "inputs": cpu_inputs,
            "real_tokens": int(real),
            "padded_tokens": int(len(chosen) * padded_len),
            "batch_size": int(len(chosen)),
            "seq_len": int(padded_len),
        })

    return batches, {
        "tokenizer_s": token_s,
        "pack_s": time.perf_counter() - pack0,
        "items": len(records),
        "min_padding_efficiency": min_eff,
        "efficiency_min_batch": eff_min_batch,
    }


def _v4_prefetched_exact_batches(
    tokenizer,
    items: list[tuple[Any, str]],
    *,
    max_length: int,
    token_budget: int,
    max_batch: int,
    pad_multiple: int,
    tokenize_window: int,
    workers: int,
    depth: int,
    min_padding_efficiency: float = 0.0,
    efficiency_min_batch: int = 16,
):
    workers = max(1, int(workers))
    depth = max(workers, int(depth))
    window = max(1, int(tokenize_window))
    windows = [items[i:i + window] for i in range(0, len(items), window)]

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oflow-v4-tok") as pool:
        futures: dict[int, Any] = {}
        submit_i = 0

        def fill():
            nonlocal submit_i
            while submit_i < len(windows) and len(futures) < depth:
                idx = submit_i
                futures[idx] = pool.submit(
                    _v4_tokenize_window,
                    tokenizer,
                    windows[idx],
                    max_length,
                    token_budget,
                    max_batch,
                    pad_multiple,
                    min_padding_efficiency,
                    efficiency_min_batch,
                )
                submit_i += 1

        fill()
        for idx in range(len(windows)):
            batches, meta = futures.pop(idx).result()
            fill()
            yield batches, meta


def _v4_sdpa_context(name: str):
    name = str(name).lower().strip()
    if name in {"", "auto", "default"}:
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        raise RuntimeError("This torch build does not expose sdpa_kernel") from exc

    mapping = {
        "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", None),
        "flash": getattr(SDPBackend, "FLASH_ATTENTION", None),
        "efficient": getattr(SDPBackend, "EFFICIENT_ATTENTION", None),
        "math": getattr(SDPBackend, "MATH", None),
    }
    backend = mapping.get(name)
    if backend is None:
        raise ValueError(f"Unsupported/unavailable SDPA backend: {name}")
    return sdpa_kernel(backend)


def _v4_model_forward(model, input_ids, attention_mask, want: str, fast_adapter: bool):
    """Specialized inference path for our multitask classifier wrappers.

    The reference wrapper computes every auxiliary head even when OFlow needs
    only decision logits or raw pooled embeddings. Skipping unused heads removes
    small kernels and keeps the GPU graph cleaner without changing the result.
    """
    if fast_adapter and all(
        hasattr(model, name)
        for name in ("encoder", "pool", "norm", "dropout", "decision_head")
    ):
        out = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled_raw = model.pool(out.last_hidden_state, attention_mask)
        if want == "embedding":
            return pooled_raw
        if want == "decision":
            pooled = model.dropout(model.norm(pooled_raw))
            return model.decision_head(pooled)
        raise ValueError(want)

    out = model(
        input_ids,
        attention_mask,
        return_embedding=True,
    )
    if want == "decision":
        return out["decision"]
    if want == "embedding":
        return out["embedding"]
    raise ValueError(want)


def _v4_gpu_exact(
    *,
    model,
    tokenizer,
    items: list[tuple[Any, str]],
    max_length: int,
    token_budget: int,
    max_batch: int,
    pad_multiple: int,
    tokenize_window: int,
    tokenizer_workers: int,
    prefetch_depth: int,
    pin_memory: bool,
    device,
    use_amp: bool,
    amp_dtype,
    want: str,
    sdpa_backend: str,
    fast_adapter: bool,
):
    import torch

    if not items:
        empty = torch.empty((0,), dtype=torch.float32)
        return [], empty, {
            "wall_s": 0.0,
            "tokenizer_task_sum_s": 0.0,
            "pack_task_sum_s": 0.0,
            "pin_memory_sum_s": 0.0,
            "h2d_cuda_ms": 0.0,
            "model_cuda_ms": 0.0,
            "d2h_wall_s": 0.0,
            "batches": 0,
            "items_total": 0,
            "real_tokens": 0,
            "padded_tokens": 0,
            "padding_efficiency": 1.0,
            "mean_batch_size": 0.0,
            "mean_seq_len": 0.0,
        }

    if hasattr(model, "eval"):
        model.eval()

    gpu_outputs = []
    output_keys = []
    live_cpu = []
    live_gpu = []
    copy_events = []
    model_events = []

    tokenizer_sum = 0.0
    pack_sum = 0.0
    pin_sum = 0.0
    real_total = 0
    padded_total = 0
    batch_sizes = []
    seq_lens = []
    item_token_counts = []

    if device.type == "cuda":
        compute_stream = torch.cuda.current_stream(device)
        copy_stream = torch.cuda.Stream(device=device)
    else:
        compute_stream = None
        copy_stream = None

    # Kania v2 policy:
    # - decision/document path keeps the proven v4 geometry;
    # - embedding/paragraph path gets a much larger tokenizer lookahead and
    #   a padding-efficiency floor. This is intentionally inferred from `want`
    #   so the existing external v5 scheduler automatically benefits too.
    is_paragraph_path = (want == "embedding")
    effective_tokenize_window = (
        max(int(tokenize_window), 8192)
        if is_paragraph_path
        else int(tokenize_window)
    )
    min_padding_efficiency = 0.88 if is_paragraph_path else 0.0
    efficiency_min_batch = 16 if is_paragraph_path else 1

    wall0 = time.perf_counter()
    with _v4_sdpa_context(sdpa_backend):
        for encoded_batches, window_meta in _v4_prefetched_exact_batches(
            tokenizer,
            items,
            max_length=max_length,
            token_budget=token_budget,
            max_batch=max_batch,
            pad_multiple=pad_multiple,
            tokenize_window=effective_tokenize_window,
            workers=tokenizer_workers,
            depth=prefetch_depth,
            min_padding_efficiency=min_padding_efficiency,
            efficiency_min_batch=efficiency_min_batch,
        ):
            tokenizer_sum += float(window_meta.get("tokenizer_s", 0.0))
            pack_sum += float(window_meta.get("pack_s", 0.0))

            for batch in encoded_batches:
                cpu_inputs = batch["inputs"]
                if device.type == "cuda" and pin_memory:
                    p0 = time.perf_counter()
                    cpu_inputs = {
                        k: (v if v.is_pinned() else v.pin_memory())
                        for k, v in cpu_inputs.items()
                    }
                    pin_sum += time.perf_counter() - p0

                real_total += int(batch["real_tokens"])
                padded_total += int(batch["padded_tokens"])
                batch_sizes.append(int(batch["batch_size"]))
                seq_lens.append(int(batch["seq_len"]))
                output_keys.extend(batch["keys"])
                item_token_counts.extend(batch.get("item_tokens", []))

                if device.type == "cuda":
                    c0 = torch.cuda.Event(enable_timing=True)
                    c1 = torch.cuda.Event(enable_timing=True)
                    ready = torch.cuda.Event()
                    with torch.cuda.stream(copy_stream):
                        c0.record(copy_stream)
                        gpu_inputs = {
                            k: v.to(device, non_blocking=pin_memory)
                            for k, v in cpu_inputs.items()
                        }
                        c1.record(copy_stream)
                        ready.record(copy_stream)
                    copy_events.append((c0, c1))
                    live_cpu.append(cpu_inputs)
                    live_gpu.append(gpu_inputs)

                    compute_stream.wait_event(ready)
                    for tensor in gpu_inputs.values():
                        tensor.record_stream(compute_stream)

                    m0 = torch.cuda.Event(enable_timing=True)
                    m1 = torch.cuda.Event(enable_timing=True)
                    m0.record(compute_stream)
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda",
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        out = _v4_model_forward(
                            model,
                            gpu_inputs["input_ids"],
                            gpu_inputs["attention_mask"],
                            want,
                            fast_adapter,
                        )
                    m1.record(compute_stream)
                    model_events.append((m0, m1))
                else:
                    gpu_inputs = {k: v.to(device) for k, v in cpu_inputs.items()}
                    with torch.inference_mode():
                        out = _v4_model_forward(
                            model,
                            gpu_inputs["input_ids"],
                            gpu_inputs["attention_mask"],
                            want,
                            fast_adapter,
                        )

                gpu_outputs.append(out)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        live_cpu.clear()
        live_gpu.clear()

    launch_wall = time.perf_counter() - wall0

    d2h0 = time.perf_counter()
    if gpu_outputs:
        merged_gpu = torch.cat(gpu_outputs, dim=0).detach()
        # Paragraph embeddings may be BF16. Keeping them BF16 for D2H halves the
        # copy volume; semantic code converts each document slice to FP32 exactly
        # as before. Decision logits are tiny, so keep reference FP32 behaviour.
        if want == "decision":
            merged = merged_gpu.float().cpu()
        else:
            merged = merged_gpu.cpu()
    else:
        merged = torch.empty((0,), dtype=torch.float32)
    d2h_wall = time.perf_counter() - d2h0

    copy_ms = 0.0
    model_ms = 0.0
    if device.type == "cuda":
        copy_ms = sum(float(a.elapsed_time(b)) for a, b in copy_events)
        model_ms = sum(float(a.elapsed_time(b)) for a, b in model_events)

    return output_keys, merged, {
        "wall_s": launch_wall + d2h_wall,
        "launch_wall_s": launch_wall,
        "tokenizer_task_sum_s": tokenizer_sum,
        "pack_task_sum_s": pack_sum,
        "pin_memory_sum_s": pin_sum,
        "h2d_cuda_ms": copy_ms,
        "model_cuda_ms": model_ms,
        "d2h_wall_s": d2h_wall,
        "batches": len(batch_sizes),
        "items_total": int(sum(batch_sizes)),
        "real_tokens": int(real_total),
        "padded_tokens": int(padded_total),
        "padding_efficiency": float(real_total / max(padded_total, 1)),
        "mean_batch_size": float(statistics.mean(batch_sizes)) if batch_sizes else 0.0,
        "max_batch_size": int(max(batch_sizes)) if batch_sizes else 0,
        "mean_seq_len": float(statistics.mean(seq_lens)) if seq_lens else 0.0,
        "max_seq_len": int(max(seq_lens)) if seq_lens else 0,
        "sdpa_backend": sdpa_backend,
        "fast_adapter": bool(fast_adapter),
        "kania_v2_paragraph_policy": bool(is_paragraph_path),
        "effective_tokenize_window": int(effective_tokenize_window),
        "min_padding_efficiency": float(min_padding_efficiency),
        "efficiency_min_batch": int(efficiency_min_batch),
        "item_token_counts": item_token_counts if want == "decision" else None,
    }


def _v4_accumulate_stage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    sums = {}
    for key in (
        "wall_s", "launch_wall_s", "tokenizer_task_sum_s", "pack_task_sum_s",
        "pin_memory_sum_s", "h2d_cuda_ms", "model_cuda_ms", "d2h_wall_s",
        "batches", "items_total", "real_tokens", "padded_tokens",
    ):
        sums[key] = sum(float(r.get(key, 0.0)) for r in rows)
    sums["batches"] = int(round(sums["batches"]))
    sums["items_total"] = int(round(sums.get("items_total", 0)))
    sums["real_tokens"] = int(round(sums["real_tokens"]))
    sums["padded_tokens"] = int(round(sums["padded_tokens"]))
    sums["padding_efficiency"] = float(
        sums["real_tokens"] / max(sums["padded_tokens"], 1)
    )
    total_batches = max(sums["batches"], 1)
    sums["mean_batch_size"] = float(
        sum(float(r.get("mean_batch_size", 0.0)) * max(int(r.get("batches", 0)), 1) for r in rows)
        / max(sum(max(int(r.get("batches", 0)), 1) for r in rows), 1)
    )
    sums["max_batch_size"] = max(int(r.get("max_batch_size", 0)) for r in rows)
    sums["global_mean_batch_size"] = float(
        sums["items_total"] / max(sums["batches"], 1)
    )
    sums["mean_seq_len"] = float(
        sum(float(r.get("mean_seq_len", 0.0)) * max(int(r.get("batches", 0)), 1) for r in rows)
        / max(sum(max(int(r.get("batches", 0)), 1) for r in rows), 1)
    )
    sums["max_seq_len"] = max(int(r.get("max_seq_len", 0)) for r in rows)
    return sums


def _v4_semantic_chunk(engine, doc_ids, doc_paragraphs, embeddings):
    t0 = time.perf_counter()
    results = []
    for doc_idx in doc_ids:
        ps = doc_paragraphs[doc_idx]
        valid = [
            (i, embeddings[(doc_idx, i)])
            for i in range(len(ps))
            if (doc_idx, i) in embeddings
        ]
        results.append((doc_idx, _semantic_fast_one_doc(engine, ps, valid)))
    return results, time.perf_counter() - t0


def oflow_attach_max(sample, args, engine, c: MaxCandidate):
    """v4 execution path.

    1) exact token-length + token-budget document batching
    2) exact token-length + token-budget paragraph batching
    3) paragraph groups are processed by document chunks
    4) semantic CPU for chunk N runs concurrently with GPU chunk N+1
    5) stage-specific SDPA backend can be autotuned
    """
    import torch

    total0 = time.perf_counter()
    (
        checkpoint_path,
        tokenizer_source,
        device,
        model,
        tokenizer,
        schema,
        train_args,
    ) = _resolve_runtime(engine, args)

    if hasattr(model, "eval"):
        model.eval()

    # Safe inference defaults. BF16 paths are unaffected by TF32, while any
    # FP32 matmuls around heads/pooling can benefit.
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    use_amp = device.type == "cuda" and (args.bf16 or args.fp16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # ------------------------------------------------------------------ docs
    doc_items = [(i, item["text"]) for i, item in enumerate(sample)]
    doc_keys, doc_logits, doc_stats = _v4_gpu_exact(
        model=model,
        tokenizer=tokenizer,
        items=doc_items,
        max_length=args.document_max_length,
        token_budget=c.doc_token_budget,
        max_batch=c.doc_max_batch,
        pad_multiple=c.pad_multiple,
        tokenize_window=c.tokenize_window,
        tokenizer_workers=c.tokenizer_workers,
        prefetch_depth=c.prefetch_depth,
        pin_memory=args.pin_memory,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        want="decision",
        sdpa_backend=c.doc_sdpa,
        fast_adapter=c.fast_adapter,
    )

    restored_logits = torch.empty((len(sample), doc_logits.shape[-1]), dtype=doc_logits.dtype)
    for pos, key in enumerate(doc_keys):
        restored_logits[int(key)] = doc_logits[pos]
    probs = torch.softmax(restored_logits.float(), dim=-1).tolist()

    # Restore exact per-document visible token counts from the already-tokenized
    # v4 batches, preserving the reference metadata schema without a second pass.
    _doc_counts_by_key = {
        int(key): int(tok)
        for key, tok in zip(doc_keys, doc_stats.get("item_token_counts") or [])
    }
    for doc_idx, (item, row_probs) in enumerate(zip(sample, probs)):
        order = sorted(range(len(row_probs)), key=lambda i: row_probs[i], reverse=True)
        tok_count = int(_doc_counts_by_key.get(doc_idx, 0))
        item["metadata"]["tokenization"] = {
            "mmbert_document_tokens_visible": tok_count,
            "mmbert_document_max_length": args.document_max_length,
            "mmbert_document_truncated": tok_count >= args.document_max_length,
        }
        item["metadata"]["old_mmbert"] = {
            "decision_probs": {
                label: float(row_probs[i]) for i, label in enumerate(schema.decisions)
            },
            "predicted": schema.decisions[order[0]],
            "confidence": float(row_probs[order[0]]),
            "second": schema.decisions[order[1]] if len(order) > 1 else None,
            "second_probability": float(row_probs[order[1]]) if len(order) > 1 else None,
            "margin": float(row_probs[order[0]] - row_probs[order[1]]) if len(order) > 1 else 1.0,
        }

    # ----------------------------------------------------------- paragraphs CPU
    build0 = time.perf_counter()
    doc_paragraphs: dict[int, list[str]] = {}
    para_items_by_doc: dict[int, list[tuple[Any, str]]] = {}
    para_total = 0
    for doc_idx, item in enumerate(sample):
        ps = engine.paragraphs(item["text"])
        if len(ps) > args.max_paragraphs_per_doc:
            n = args.max_paragraphs_per_doc
            chosen_idx = sorted(
                set(round(i * (len(ps) - 1) / max(n - 1, 1)) for i in range(n))
            )
            ps = [ps[i] for i in chosen_idx]
        doc_paragraphs[doc_idx] = ps
        rows = []
        for para_idx, p in enumerate(ps):
            if len(engine.words(p)) >= 3:
                rows.append(((doc_idx, para_idx), p))
                para_total += 1
        para_items_by_doc[doc_idx] = rows
    paragraph_build_s = time.perf_counter() - build0

    # ------------------------------------------------------ para GPU / sem CPU
    chunk_size = max(1, int(c.semantic_chunk_docs))
    doc_chunks = [
        list(range(i, min(i + chunk_size, len(sample))))
        for i in range(0, len(sample), chunk_size)
    ]

    para_stage_rows = []
    semantic_futures = []
    semantic_task_sum = 0.0
    semantic_overlap_start = None

    sem_pool = ThreadPoolExecutor(
        max_workers=max(1, int(c.semantic_workers)),
        thread_name_prefix="oflow-v4-sem",
    )
    try:
        for chunk_doc_ids in doc_chunks:
            chunk_items = []
            for doc_idx in chunk_doc_ids:
                chunk_items.extend(para_items_by_doc[doc_idx])

            para_keys, para_embs, para_stats = _v4_gpu_exact(
                model=model,
                tokenizer=tokenizer,
                items=chunk_items,
                max_length=args.paragraph_max_length,
                token_budget=c.para_token_budget,
                max_batch=c.para_max_batch,
                pad_multiple=c.pad_multiple,
                tokenize_window=c.tokenize_window,
                tokenizer_workers=c.tokenizer_workers,
                prefetch_depth=c.prefetch_depth,
                pin_memory=args.pin_memory,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                want="embedding",
                sdpa_backend=c.para_sdpa,
                fast_adapter=c.fast_adapter,
            )
            para_stage_rows.append(para_stats)
            embeddings = {
                key: emb for key, emb in zip(para_keys, para_embs)
            }

            if semantic_overlap_start is None:
                semantic_overlap_start = time.perf_counter()
            semantic_futures.append(
                sem_pool.submit(
                    _v4_semantic_chunk,
                    engine,
                    chunk_doc_ids,
                    doc_paragraphs,
                    embeddings,
                )
            )

        # GPU has now finished all paragraph chunks. Only the tail of semantic
        # CPU that was not hidden underneath later GPU work remains exposed.
        wait0 = time.perf_counter()
        for fut in semantic_futures:
            sem_results, sem_s = fut.result()
            semantic_task_sum += float(sem_s)
            for doc_idx, sem in sem_results:
                sample[doc_idx]["metadata"]["semantic_structure"] = sem
        semantic_exposed_wait_s = time.perf_counter() - wait0
    finally:
        sem_pool.shutdown(wait=True)

    paragraph_stats = _v4_accumulate_stage(para_stage_rows)
    semantic_overlap_wall_s = (
        time.perf_counter() - semantic_overlap_start
        if semantic_overlap_start is not None else 0.0
    )
    total_s = time.perf_counter() - total0

    return {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "tokenizer": str(tokenizer_source),
        "device": str(device),
        "decisions": schema.decisions,
        "oflow_stats": {
            "total_s": total_s,
            "document_pass": doc_stats,
            "paragraph_build_s": paragraph_build_s,
            "paragraph_pass": paragraph_stats,
            "semantic_cpu_task_sum_s": semantic_task_sum,
            "semantic_exposed_wait_s": semantic_exposed_wait_s,
            "semantic_overlap_wall_s": semantic_overlap_wall_s,
            "semantic_hidden_fraction": float(
                max(0.0, min(1.0, 1.0 - semantic_exposed_wait_s / max(semantic_task_sum, 1e-9)))
            ),
            "paragraphs_total": para_total,
            "candidate": asdict(c),
        },
    }


def run_oflow_max(engine, sample, args, c: MaxCandidate):
    work = copy.deepcopy(sample)
    cuda_sync()
    t0 = time.perf_counter()
    info = oflow_attach_max(work, args, engine, c)
    cuda_sync()
    dt = time.perf_counter() - t0
    return work, dt, info["oflow_stats"]


def _v4_run_candidate(engine, sample, args, c, reference_work, reference_dps, approx_tok):
    rows = []
    last_work = None
    last_stats = None
    for r in range(args.repeats):
        work, dt, stats = run_oflow_max(engine, sample, args, c)
        last_work = work
        last_stats = stats
        rows.append({
            "repeat": r + 1,
            "seconds": dt,
            "docs_per_second": len(sample) / max(dt, 1e-9),
            "approx_tokens_per_second": approx_tok / max(dt, 1e-9),
        })
    dps = statistics.median(x["docs_per_second"] for x in rows)
    eq = compare_semantic(reference_work, last_work)
    return {
        "candidate": asdict(c),
        "label": c.label,
        "seconds": statistics.median(x["seconds"] for x in rows),
        "docs_per_second": dps,
        "approx_tokens_per_second": statistics.median(x["approx_tokens_per_second"] for x in rows),
        "speedup_vs_reference": dps / max(reference_dps, 1e-9),
        "equivalence": eq,
        "stats": last_stats,
        "runs": rows,
    }


def _v4_print_result(prefix: str, idx: int, result: dict[str, Any]):
    eq = result.get("equivalence", {})
    print(
        f"  [{prefix}{idx:02d}] {result['label']:92s} -> "
        f"{result['docs_per_second']:7.2f} docs/s  "
        f"{result['speedup_vs_reference']:5.2f}x  "
        f"equiv={'YES' if eq.get('equivalent') else 'NO'}  "
        f"maxΔ={eq.get('max_abs_float_diff', 0.0):.2e}"
    )


def _v4_clone(c: MaxCandidate, **changes) -> MaxCandidate:
    d = asdict(c)
    d.update(changes)
    return MaxCandidate(**d)


def main_v4():
    args = parse_args_v4()
    here = Path(__file__).resolve().parent

    input_path = resolve_path(args.input, here)
    engine_path = resolve_path(args.engine_script, here)
    pipeline_path = resolve_path(args.pipeline_script, here)
    checkpoint = resolve_path(args.checkpoint, here)
    tokenizer = Path(args.tokenizer)
    if args.tokenizer and not tokenizer.exists():
        q = here / tokenizer
        if q.exists():
            tokenizer = q.resolve()

    for pth in (input_path, engine_path, pipeline_path, checkpoint):
        if not pth.exists():
            raise FileNotFoundError(pth)

    args.input = str(input_path)
    args.checkpoint = str(checkpoint)
    args.tokenizer = str(tokenizer) if tokenizer.exists() else args.tokenizer
    args.pin_memory = bool(args.pin_memory)

    print("=" * 118)
    print("OFlow v4 MAX — TOKEN-BUDGET / OVERLAP / SDPA AUTOTUNE")
    print("=" * 118)

    system = basic_system_probe()
    print("[system]")
    for k, v in system.items():
        print(f"  {k:24s}: {v}")

    probes = {}
    if args.full_probe:
        print("\n[full-probe] storage...")
        probes["storage"] = storage_probe(input_path, args.probe_mb, args.gzip_probe_records)
        for k, v in probes["storage"].items():
            print(f"  {k:24s}: {v}")
        print("[full-probe] H2D / D2H...")
        probes["transfer"] = full_transfer_probe(args.probe_mb)
        for k, v in probes["transfer"].items():
            print(f"  {k:24s}: {v}")

    engine = import_file("oflow_v4_selector", engine_path)
    pipeline = import_file("oflow_v4_pipeline", pipeline_path)
    if args.backend == "oris":
        if not hasattr(pipeline, "make_oris_filter_loader"):
            raise RuntimeError("ORIS pipeline script has no make_oris_filter_loader()")
        engine.load_mmbert = pipeline.make_oris_filter_loader()

    patch = pipeline.PersistentTorchPatch(engine, checkpoint)
    try:
        args.max_scan = int(args.max_scan)
        args.sample_size = int(args.sample_docs)
        args.text_key = ""

        print("\n[sample] reservoir + surface metadata...")
        sample, scan, surface_s = prepare_sample(engine, input_path, args)
        approx_tok = approx_sample_tokens(sample)
        print(
            f"  docs={len(sample):,} approx_tokens={approx_tok:,} "
            f"surface={surface_s:.3f}s ({len(sample)/max(surface_s,1e-9):.1f} docs/s)"
        )

        warm = copy.deepcopy(sample[: max(1, min(args.warmup_docs, len(sample)))])
        args.document_batch_size = min(8, len(warm))
        args.paragraph_batch_size = 64
        print("\n[warmup] reference model/tokenizer/CUDA...")
        cuda_sync()
        w0 = time.perf_counter()
        engine.attach_mmbert_metadata(warm, args)
        cuda_sync()
        print(f"  {time.perf_counter()-w0:.3f}s")

        # ------------------------------------------------------------- reference
        bdoc, bpara = parse_pair(args.baseline_config)
        print("\n[reference] current sequential selector path...")
        reference_runs = []
        reference_work = None
        for r in range(args.repeats):
            work, dt = run_reference(engine, sample, args, bdoc, bpara)
            reference_work = work
            row = {
                "repeat": r + 1,
                "seconds": dt,
                "docs_per_second": len(sample) / max(dt, 1e-9),
                "approx_tokens_per_second": approx_tok / max(dt, 1e-9),
            }
            reference_runs.append(row)
            print(
                f"  repeat={r+1} {row['docs_per_second']:.2f} docs/s "
                f"~{row['approx_tokens_per_second']:,.0f} tok/s ({dt:.3f}s)"
            )
        reference_dps = statistics.median(x["docs_per_second"] for x in reference_runs)

        all_results = []

        # ---------------------------------------------------------- legacy anchor
        if args.legacy_anchor:
            print("\n[anchor] previous v3 bounded-char bucket...")
            legacy = Candidate(16, 64, 2, 4, 1, "bucket")
            try:
                rows = []
                last_work = None
                last_stats = None
                for r in range(args.repeats):
                    work, dt, stats = run_oflow(engine, sample, args, legacy, args.pin_memory)
                    last_work, last_stats = work, stats
                    rows.append({
                        "repeat": r + 1,
                        "seconds": dt,
                        "docs_per_second": len(sample) / max(dt, 1e-9),
                    })
                dps = statistics.median(x["docs_per_second"] for x in rows)
                eq = compare_semantic(reference_work, last_work)
                anchor = {
                    "phase": "legacy_anchor",
                    "label": "v3 bucket doc=16 para=64",
                    "docs_per_second": dps,
                    "speedup_vs_reference": dps / max(reference_dps, 1e-9),
                    "equivalence": eq,
                    "stats": last_stats,
                    "runs": rows,
                }
                all_results.append(anchor)
                print(
                    f"  v3 bucket -> {dps:.2f} docs/s  {anchor['speedup_vs_reference']:.2f}x "
                    f"equiv={'YES' if eq.get('equivalent') else 'NO'}"
                )
            except Exception as exc:
                print(f"  legacy anchor FAILED: {exc!r}")

        # --------------------------------------------------------- phase 1 shape
        base_common = dict(
            tokenizer_workers=args.tokenizer_workers,
            prefetch_depth=args.prefetch_depth,
            tokenize_window=args.tokenize_window,
            semantic_chunk_docs=args.semantic_chunk_docs,
            semantic_workers=args.semantic_workers,
            pad_multiple=args.pad_multiple,
            doc_sdpa="auto",
            para_sdpa="auto",
            fast_adapter=True,
        )
        shape_candidates = [
            MaxCandidate(
                doc_token_budget=a,
                doc_max_batch=b,
                para_token_budget=c,
                para_max_batch=d,
                **base_common,
            )
            for a, b, c, d in _v4_parse_shape_presets(args.shape_presets)
        ]

        print("\n[phase 1] exact token-budget shapes...")
        shape_results = []
        for idx, c in enumerate(shape_candidates, 1):
            try:
                result = _v4_run_candidate(
                    engine, sample, args, c, reference_work, reference_dps, approx_tok
                )
                result["phase"] = "shape"
                shape_results.append(result)
                all_results.append(result)
                _v4_print_result("S", idx, result)
            except Exception as exc:
                print(f"  [S{idx:02d}] {c.label} -> FAILED: {exc!r}")

        valid_shape = [
            x for x in shape_results
            if x.get("equivalence", {}).get("equivalent")
        ]
        if not valid_shape:
            raise RuntimeError("No equivalent v4 shape candidate")
        best = max(valid_shape, key=lambda x: x["docs_per_second"])
        best_c = MaxCandidate(**best["candidate"])

        # -------------------------------------------------------- phase 2 kernel
        backends = [x.strip().lower() for x in args.kernel_backends.split(",") if x.strip()]
        print("\n[phase 2] SDPA kernel backends on winning shape...")
        kernel_results = []
        for idx, backend in enumerate(backends, 1):
            c = _v4_clone(best_c, doc_sdpa=backend, para_sdpa=backend)
            try:
                result = _v4_run_candidate(
                    engine, sample, args, c, reference_work, reference_dps, approx_tok
                )
                result["phase"] = "kernel"
                kernel_results.append(result)
                all_results.append(result)
                _v4_print_result("K", idx, result)
            except Exception as exc:
                print(f"  [K{idx:02d}] sdpa={backend} -> FAILED: {exc!r}")

        valid_kernel = [x for x in kernel_results if x.get("equivalence", {}).get("equivalent")]
        if valid_kernel:
            # Stage-specific mixed backend: choose the backend that minimized each
            # stage CUDA time, then benchmark the combined configuration.
            doc_best = min(
                valid_kernel,
                key=lambda x: float(x.get("stats", {}).get("document_pass", {}).get("model_cuda_ms", 1e30)),
            )
            para_best = min(
                valid_kernel,
                key=lambda x: float(x.get("stats", {}).get("paragraph_pass", {}).get("model_cuda_ms", 1e30)),
            )
            mixed = _v4_clone(
                best_c,
                doc_sdpa=doc_best["candidate"]["doc_sdpa"],
                para_sdpa=para_best["candidate"]["para_sdpa"],
            )
            print("\n[phase 2b] mixed stage-specific SDPA...")
            try:
                result = _v4_run_candidate(
                    engine, sample, args, mixed, reference_work, reference_dps, approx_tok
                )
                result["phase"] = "kernel_mixed"
                all_results.append(result)
                _v4_print_result("M", 1, result)
                valid_kernel.append(result)
            except Exception as exc:
                print(f"  [M01] mixed SDPA -> FAILED: {exc!r}")

            candidate_pool = valid_shape + valid_kernel
            best = max(candidate_pool, key=lambda x: x["docs_per_second"])
            best_c = MaxCandidate(**best["candidate"])

        # ----------------------------------------------------------- phase 3 pad
        pad_values = sorted({int(x) for x in args.pad_search.split(",") if x.strip()})
        print("\n[phase 3] padding multiple around current winner...")
        pad_results = []
        for idx, pad in enumerate(pad_values, 1):
            c = _v4_clone(best_c, pad_multiple=pad)
            try:
                result = _v4_run_candidate(
                    engine, sample, args, c, reference_work, reference_dps, approx_tok
                )
                result["phase"] = "pad"
                pad_results.append(result)
                all_results.append(result)
                _v4_print_result("P", idx, result)
            except Exception as exc:
                print(f"  [P{idx:02d}] pad={pad} -> FAILED: {exc!r}")
        valid_pad = [x for x in pad_results if x.get("equivalence", {}).get("equivalent")]
        if valid_pad:
            best = max([best] + valid_pad, key=lambda x: x["docs_per_second"])
            best_c = MaxCandidate(**best["candidate"])

        # --------------------------------------------------------- phase 4 chunk
        chunks = sorted({int(x) for x in args.chunk_search.split(",") if x.strip()})
        print("\n[phase 4] semantic/GPU overlap chunk size...")
        chunk_results = []
        for idx, chunk in enumerate(chunks, 1):
            c = _v4_clone(best_c, semantic_chunk_docs=chunk)
            try:
                result = _v4_run_candidate(
                    engine, sample, args, c, reference_work, reference_dps, approx_tok
                )
                result["phase"] = "chunk"
                chunk_results.append(result)
                all_results.append(result)
                _v4_print_result("C", idx, result)
            except Exception as exc:
                print(f"  [C{idx:02d}] chunk={chunk} -> FAILED: {exc!r}")
        valid_chunk = [x for x in chunk_results if x.get("equivalence", {}).get("equivalent")]
        if valid_chunk:
            best = max([best] + valid_chunk, key=lambda x: x["docs_per_second"])
            best_c = MaxCandidate(**best["candidate"])

        # ---------------------------------------------------------------- result
        print("\n" + "=" * 118)
        print("OFLOW v4 MAX RESULT")
        print("=" * 118)
        print(f"Reference : {reference_dps:.2f} docs/s")
        print(
            f"Best      : {best['docs_per_second']:.2f} docs/s | "
            f"{best['speedup_vs_reference']:.3f}x"
        )
        print(f"Config    : {best_c.label}")
        print(f"Equivalent: {best.get('equivalence', {}).get('equivalent')}")

        st = best.get("stats", {})
        doc = st.get("document_pass", {})
        para = st.get("paragraph_pass", {})
        print("\n[best telemetry]")
        print(
            f"  doc CUDA      : {doc.get('model_cuda_ms', 0.0):.1f} ms | "
            f"pad-eff={doc.get('padding_efficiency', 0.0):.1%} | "
            f"batches={doc.get('batches', 0)} meanB={doc.get('mean_batch_size', 0.0):.1f}"
        )
        print(
            f"  para CUDA     : {para.get('model_cuda_ms', 0.0):.1f} ms | "
            f"pad-eff={para.get('padding_efficiency', 0.0):.1%} | "
            f"batches={para.get('batches', 0)} meanB={para.get('mean_batch_size', 0.0):.1f}"
        )
        print(
            f"  doc tokenizer : {doc.get('tokenizer_task_sum_s', 0.0):.3f}s | "
            f"pack={doc.get('pack_task_sum_s', 0.0):.3f}s"
        )
        print(
            f"  para tokenizer: {para.get('tokenizer_task_sum_s', 0.0):.3f}s | "
            f"pack={para.get('pack_task_sum_s', 0.0):.3f}s"
        )
        print(f"  paragraph build: {st.get('paragraph_build_s', 0.0):.3f}s")
        print(
            f"  semantic CPU task: {st.get('semantic_cpu_task_sum_s', 0.0):.3f}s | "
            f"exposed={st.get('semantic_exposed_wait_s', 0.0):.3f}s | "
            f"hidden={st.get('semantic_hidden_fraction', 0.0):.1%}"
        )

        # Save a ready-to-use manual config for the future product wrapper.
        manual_config = {
            "oflow_config_version": 1,
            "backend": args.backend,
            "document_max_length": args.document_max_length,
            "paragraph_max_length": args.paragraph_max_length,
            **asdict(best_c),
        }
        config_path = Path("oflow_best_config.json")
        config_path.write_text(
            json.dumps(manual_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = {
            "oflow_version": "v4-max",
            "backend": args.backend,
            "input": str(input_path),
            "checkpoint": str(checkpoint),
            "system": system,
            "full_probe": probes,
            "sample": {
                "documents": len(sample),
                "approx_tokens": approx_tok,
                "scan": scan,
                "surface_seconds": surface_s,
            },
            "reference": {
                "document_batch": bdoc,
                "paragraph_batch": bpara,
                "median_docs_per_second": reference_dps,
                "runs": reference_runs,
            },
            "all_results": all_results,
            "best": best,
            "best_manual_config": manual_config,
        }
        out = Path(args.output)
        if not out.is_absolute():
            out = Path.cwd() / out
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nManual config: {config_path.resolve()}")
        print(f"JSON report  : {out.resolve()}")
        print("=" * 118)
    finally:
        patch.close()


if __name__ == "__main__":
    main_v4()
