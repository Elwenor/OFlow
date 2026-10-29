"""ORIS/mmBERT semantic geometry preserved from supplied OFlow v5 beta.

Extracted unchanged except function name; operates on CPU-owned embeddings.
Reference selector equivalence needs the original external environment.
"""


def semantic_one_doc(engine, ps, valid):
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
        ctx = torch.where(
            den > 1e-12, (left * right).sum(dim=1) / den.clamp_min(1e-12), torch.zeros_like(den)
        )
        for j, simv in enumerate(ctx.tolist(), start=1):
            contextual.append(
                {
                    "after_paragraph": ids[j - 1],
                    "before_paragraph": ids[j],
                    "context_cosine": float(simv),
                    "context_boundary_score": 1.0 - float(simv),
                }
            )

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
        chars.append(c)
        word_counts.append(w)
        toks.append(t)
        para_mass[pid] = {"chars": c, "words": w, "tokens": t}

    def pref(xs):
        out = [0]
        z = 0
        for x in xs:
            z += int(x)
            out.append(z)
        return out

    pc, pw, pt = pref(chars), pref(word_counts), pref(toks)
    total_chars, total_words, total_tokens = pc[-1], pw[-1], pt[-1]

    def mass(a, b):
        c = pc[b] - pc[a]
        w = pw[b] - pw[a]
        t = pt[b] - pt[a]
        return {
            "paragraphs": b - a,
            "chars": c,
            "words": w,
            "tokens": t,
            "paragraph_ratio": (b - a) / max(n, 1),
            "char_ratio": c / max(total_chars, 1),
            "word_ratio": w / max(total_words, 1),
            "token_ratio": t / max(total_tokens, 1),
        }

    min_split_side_words = (
        max(100, min(150, int(round(0.20 * total_words)))) if total_words else 100
    )
    min_split_token_ratio = 0.20
    split_candidates = []
    if n >= 2:
        sim_prefix = [0.0]
        z = 0.0
        for x in sims:
            z += float(x)
            sim_prefix.append(z)
        for cut in range(1, n):
            lm, rm = mass(0, cut), mass(cut, n)
            if (
                lm["token_ratio"] < min_split_token_ratio
                or rm["token_ratio"] < min_split_token_ratio
                or lm["words"] < min_split_side_words
                or rm["words"] < min_split_side_words
            ):
                continue
            lcnt = max(cut - 1, 0)
            rcnt = max(n - cut - 1, 0)
            left_coh = (sim_prefix[cut - 1] / lcnt) if lcnt else 0.0
            right_coh = ((sim_prefix[n - 1] - sim_prefix[cut]) / rcnt) if rcnt else 0.0
            left_mean = (prefix[cut] - prefix[0]) / float(cut)
            right_mean = (prefix[n] - prefix[cut]) / float(n - cut)
            den = torch.linalg.vector_norm(left_mean) * torch.linalg.vector_norm(right_mean)
            cross = float(torch.dot(left_mean, right_mean) / den) if float(den) > 1e-12 else 0.0
            token_balance = 2.0 * min(lm["token_ratio"], rm["token_ratio"])
            lq = max(0.0, min(1.0, (left_coh - 0.74) / 0.16))
            rq = max(0.0, min(1.0, (right_coh - 0.74) / 0.16))
            sq = max(0.0, min(1.0, (0.78 - cross) / 0.30))
            bq = max(0.0, min(1.0, token_balance))
            score = min(lq, rq) * sq * bq
            split_candidates.append(
                {
                    "after_paragraph": ids[cut - 1],
                    "before_paragraph": ids[cut],
                    "left_paragraphs": cut,
                    "right_paragraphs": n - cut,
                    "left_chars": int(lm["chars"]),
                    "right_chars": int(rm["chars"]),
                    "left_words": int(lm["words"]),
                    "right_words": int(rm["words"]),
                    "left_tokens": int(lm["tokens"]),
                    "right_tokens": int(rm["tokens"]),
                    "left_paragraph_ratio": lm["paragraph_ratio"],
                    "right_paragraph_ratio": rm["paragraph_ratio"],
                    "left_char_ratio": lm["char_ratio"],
                    "right_char_ratio": rm["char_ratio"],
                    "left_word_ratio": lm["word_ratio"],
                    "right_word_ratio": rm["word_ratio"],
                    "left_token_ratio": lm["token_ratio"],
                    "right_token_ratio": rm["token_ratio"],
                    "min_required_side_words": min_split_side_words,
                    "min_required_side_token_ratio": min_split_token_ratio,
                    "left_coherence": left_coh,
                    "right_coherence": right_coh,
                    "cross_region_cosine": cross,
                    "token_balance": token_balance,
                    "left_quality": lq,
                    "right_quality": rq,
                    "separation_quality": sq,
                    "recoverable_split_score": score,
                }
            )
    best_split = max(split_candidates, key=lambda x: x["recoverable_split_score"], default=None)

    region_threshold = 0.74
    ranges = []
    start = 0
    for edge, sim in enumerate(sims):
        if sim < region_threshold:
            ranges.append((start, edge + 1))
            start = edge + 1
    ranges.append((start, n))
    regions = []
    sim_prefix = [0.0]
    z = 0.0
    for x in sims:
        z += float(x)
        sim_prefix.append(z)
    for a, b in ranges:
        m = mass(a, b)
        cnt = max(b - a - 1, 0)
        coh = ((sim_prefix[b - 1] - sim_prefix[a]) / cnt) if cnt else None
        regions.append(
            {
                "start_paragraph": ids[a],
                "end_paragraph": ids[b - 1],
                "paragraphs": int(m["paragraphs"]),
                "chars": int(m["chars"]),
                "words": int(m["words"]),
                "tokens": int(m["tokens"]),
                "paragraph_ratio": m["paragraph_ratio"],
                "char_ratio": m["char_ratio"],
                "word_ratio": m["word_ratio"],
                "token_ratio": m["token_ratio"],
                "ratio": m["token_ratio"],
                "coherence": coh,
            }
        )
    rs = sorted(regions, key=lambda r: r["token_ratio"], reverse=True)
    largest = float(rs[0]["token_ratio"]) if rs else 0.0
    second = float(rs[1]["token_ratio"]) if len(rs) > 1 else 0.0
    substantial = [
        r
        for r in regions
        if r["token_ratio"] >= 0.20
        and r["words"] >= min_split_side_words
        and (r["coherence"] is None or float(r["coherence"]) >= 0.74)
    ]

    return {
        "paragraphs_embedded": n,
        "coherence_observable": len(sims) > 0,
        "adjacent_cosine_mean": mean_adj,
        "adjacent_cosine_min": min_adj,
        "adjacent_boundary_max": (1.0 - min_adj) if min_adj is not None else None,
        "context_boundary_mean": sum(boundary_scores) / len(boundary_scores)
        if boundary_scores
        else 0.0,
        "context_boundary_max": max(boundary_scores, default=0.0),
        "tail_anomaly": tail_anomaly,
        "best_boundary": best,
        "largest_coherent_region_ratio": largest,
        "second_coherent_region_ratio": second,
        "substantial_region_count": len(substantial),
        "substantial_region_coverage": sum(float(r["token_ratio"]) for r in substantial),
        "semantic_regions": regions,
        "region_threshold": region_threshold,
        "region_size_basis": "token_mass",
        "total_embedded_chars": total_chars,
        "total_embedded_words": total_words,
        "total_embedded_tokens": total_tokens,
        "min_split_side_words": min_split_side_words,
        "min_split_token_ratio": min_split_token_ratio,
        "best_recoverable_split": best_split,
        "recoverable_split_score": float(best_split["recoverable_split_score"])
        if best_split
        else 0.0,
        "adjacent_profile": adjacent,
        "context_boundary_profile": contextual,
        "recoverable_split_profile": split_candidates,
    }
