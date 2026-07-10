"""Second-round diagnostics, added after the first review of the project.

    bootstrap_ci               — percentile bootstrap CI over per-pair values
    report_verb_continuation   — vc baseline with CIs on arbitrary pair sets  (step 09)
    run_attention_analysis     — attention at the disambiguator → V-amb word  (step 10)
    run_patching_sweep         — activation patching normal→GPS               (step 11)
    collect_head_means / HeadsMeanAblated / run_mean_ablation
                               — mean ablation replication of key results     (step 12)
    run_null_distribution      — random single-head ablation null             (step 13)

Model-agnostic, like `_ablation.py`: the architecture is read from the
config. Lives next to `_ablation.py` so both qwen4b and qwen14b scripts
import it the same way.
"""

from __future__ import annotations

import numpy as np
import torch
from _ablation import (
    HeadsAblated,
    _v_amb_word,
    get_architecture,
    measure_effect,
    measure_lexical_baseline,
    measure_verb_continuation,
)
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_finder.arch import decoder_layers
from model_finder.models import ModelSpec
from model_finder.sentences import PAIRS, Pair, by_template
from model_finder.surprisal import _DTYPES, surprisal

# ──────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────


def bootstrap_ci(
    values: list[float], n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile-bootstrap CI of the mean, NaN-safe."""
    a = np.array([v for v in values if not np.isnan(v)])
    if a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ──────────────────────────────────────────────────────────────────────────
# Step 09 — verb-continuation baseline with CIs, canonical vs novel
# ──────────────────────────────────────────────────────────────────────────


def report_verb_continuation(
    model, tokenizer, pairs: list[Pair], device: str, label: str
) -> dict:
    """measure_verb_continuation + bootstrap CIs + per-pair table."""
    summary, pair_results = measure_verb_continuation(model, tokenizer, pairs, device)
    ci_gps = bootstrap_ci([r["s_vc_gps"] for r in pair_results])
    ci_norm = bootstrap_ci([r["s_vc_normal"] for r in pair_results])
    ci_delta = bootstrap_ci([r["vc_delta"] for r in pair_results])

    print(f"  {label} (n={len(pair_results)}):")
    print(
        f"    s_vc_gps    = {summary['mean_s_vc_gps']:6.3f}   "
        f"95% CI [{ci_gps[0]:.3f}, {ci_gps[1]:.3f}]"
    )
    print(
        f"    s_vc_normal = {summary['mean_s_vc_normal']:6.3f}   "
        f"95% CI [{ci_norm[0]:.3f}, {ci_norm[1]:.3f}]"
    )
    print(
        f"    vc_delta    = {summary['mean_vc_delta']:+6.3f}   "
        f"95% CI [{ci_delta[0]:+.3f}, {ci_delta[1]:+.3f}]"
    )
    print()
    print(f"    {'id':>3}  {'vc':<12}  {'s_vc_gps':>9}  {'s_vc_norm':>9}  {'vc_delta':>9}")
    print("    " + "-" * 50)
    for r in pair_results:
        print(
            f"    {r['id']:>3}  {r['verb_continuation']:<12}  "
            f"{r['s_vc_gps']:>9.3f}  {r['s_vc_normal']:>9.3f}  {r['vc_delta']:>+9.3f}"
        )
    print()

    return {
        "label": label,
        "summary": summary,
        "ci95": {"s_vc_gps": ci_gps, "s_vc_normal": ci_norm, "vc_delta": ci_delta},
        "pair_results": pair_results,
    }


def report_lexical_baseline(
    model, tokenizer, pairs: list[Pair], device: str, label: str
) -> dict:
    """measure_lexical_baseline + bootstrap CI on `full − stripped` + table."""
    summary, pair_results = measure_lexical_baseline(model, tokenizer, pairs, device)
    fs = [r["s_full"] - r["s_stripped"] for r in pair_results]
    ci_fs = bootstrap_ci(fs)

    print(f"  {label} (n={len(pair_results)}):")
    print(f"    s_full     = {summary['mean_s_full']:.3f}")
    print(f"    s_stripped = {summary['mean_s_stripped']:.3f}")
    print(f"    s_anchor   = {summary['mean_s_anchor']:.3f}")
    print(
        f"    full − stripped = {summary['mean_s_full'] - summary['mean_s_stripped']:+.3f}   "
        f"95% CI [{ci_fs[0]:+.3f}, {ci_fs[1]:+.3f}]"
    )
    print()
    header = (
        f"    {'id':>3}  {'v_amb':<10}  {'vc':<12}  "
        f"{'s_full':>7}  {'s_strip':>7}  {'s_anch':>7}  {'full-strip':>10}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    for r in pair_results:
        print(
            f"    {r['id']:>3}  {r['v_amb_word']:<10}  {r['verb_continuation']:<12}  "
            f"{r['s_full']:>7.3f}  {r['s_stripped']:>7.3f}  {r['s_anchor']:>7.3f}  "
            f"{r['s_full'] - r['s_stripped']:>+10.3f}"
        )
    print()

    return {
        "label": label,
        "summary": summary,
        "ci95_full_minus_stripped": ci_fs,
        "pair_results": pair_results,
    }


# ──────────────────────────────────────────────────────────────────────────
# Step 10 — attention patterns at the disambiguator
# ──────────────────────────────────────────────────────────────────────────


def load_model_eager(device: str, spec: ModelSpec):
    """Like model_finder.surprisal.load_model but with eager attention,
    which is required for `output_attentions=True`."""
    print(f"  loading {spec.hf_name} (dtype={spec.dtype}, attn=eager) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name,
        dtype=_DTYPES[spec.dtype],
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _word_token_indices(
    tokenizer, text: str, char_start: int, char_end: int
) -> tuple[list[int], list[int]]:
    # Overlap-based match: BPE offset mappings include the token's leading
    # space (" man" spans the space too), so an exact containment test on the
    # word's own chars silently matches nothing.
    enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    idx = [
        i
        for i, (a, b) in enumerate(enc["offset_mapping"])
        if a < char_end and b > char_start and b > a
    ]
    return enc["input_ids"], idx


def _attention_to_vamb(
    model, tokenizer, prefix: str, critical_word: str, device: str
) -> np.ndarray:
    """Attention weight from the disambiguator (last critical-word token) back
    to the V-amb word (last word of `prefix`), per (layer, head).

    Returns array (n_layers, n_heads); the weight is summed over the V-amb
    word's subword tokens."""
    v_amb = prefix.split()[-1]
    text = prefix + " " + critical_word
    ids, v_idx = _word_token_indices(
        tokenizer, text, len(prefix) - len(v_amb), len(prefix)
    )
    _, c_idx = _word_token_indices(
        tokenizer, text, len(prefix) + 1, len(prefix) + 1 + len(critical_word)
    )
    if not v_idx or not c_idx:
        n_layers, n_heads, _ = get_architecture(model)
        return np.full((n_layers, n_heads), np.nan)
    q = c_idx[-1]

    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    # attentions: tuple of n_layers × (1, n_heads, T, T)
    scores = [att[0, :, q, v_idx].sum(dim=-1).float().cpu().numpy() for att in out.attentions]
    return np.stack(scores)  # (n_layers, n_heads)


def run_attention_analysis(
    model, tokenizer, pairs: list[Pair], device: str, top_n: int = 15
) -> dict:
    """For every NV pair, measure attention disambiguator→V-amb in the GPS and
    NORMAL conditions, plus per-pair vc_delta for correlation.

    The hypothesised "retrospective read" head should (a) attend strongly to
    the V-amb word at the disambiguator, and (b) do so *more* in GPS than in
    NORMAL, and (c) its per-pair attention should correlate with lower
    residual comprehension cost (vc_delta)."""
    nv = [p for p in pairs if p.template == "NV" and p.verb_continuation]
    n_layers, n_heads, _ = get_architecture(model)

    per_pair = []
    gps_mats, norm_mats = [], []
    for p in nv:
        a_gps = _attention_to_vamb(model, tokenizer, p.gps_prefix, p.critical_word, device)
        a_norm = _attention_to_vamb(model, tokenizer, p.normal_prefix, p.critical_word, device)
        full_gps = p.gps_prefix + " " + p.critical_word
        full_norm = p.normal_prefix + " " + p.critical_word
        s_gps = surprisal(model, tokenizer, full_gps, p.verb_continuation, device)
        s_norm = surprisal(model, tokenizer, full_norm, p.verb_continuation, device)
        gps_mats.append(a_gps)
        norm_mats.append(a_norm)
        per_pair.append(
            {"id": p.id, "v_amb": _v_amb_word(p), "vc_delta": s_gps - s_norm}
        )
        print(f"    pair {p.id:>3} ({_v_amb_word(p)}) done", flush=True)

    gps_stack = np.stack(gps_mats)   # (n_pairs, L, H)
    norm_stack = np.stack(norm_mats)
    mean_gps = np.nanmean(gps_stack, axis=0)
    mean_diff = np.nanmean(gps_stack - norm_stack, axis=0)
    vc_deltas = np.array([r["vc_delta"] for r in per_pair])

    def _top(matrix: np.ndarray, key: str) -> list[dict]:
        flat = [
            (int(li), int(h), float(matrix[li, h]))
            for li in range(n_layers)
            for h in range(n_heads)
            if not np.isnan(matrix[li, h])
        ]
        flat.sort(key=lambda t: abs(t[2]), reverse=True)
        rows = []
        print(f"\n  Top {top_n} heads by |{key}| (attention disambiguator → V-amb word):")
        print(
            f"    {'head':>8}  {'attn_gps':>9}  {'attn_norm':>9}  {'gps-norm':>9}  "
            f"{'spearman(attn_gps, vc_delta)':>28}"
        )
        print("    " + "-" * 72)
        for li, h, _val in flat[:top_n]:
            series = gps_stack[:, li, h]
            ok = ~np.isnan(series) & ~np.isnan(vc_deltas)
            if ok.sum() >= 3 and np.std(series[ok]) > 0:
                rho, pval = stats.spearmanr(series[ok], vc_deltas[ok])
            else:
                rho, pval = float("nan"), float("nan")
            row = {
                "layer": li,
                "head": h,
                "mean_attn_gps": float(mean_gps[li, h]),
                "mean_attn_diff": float(mean_diff[li, h]),
                "spearman_vc_delta": float(rho),
                "spearman_p": float(pval),
            }
            rows.append(row)
            print(
                f"    L{li:>2}.H{h:<3}  {mean_gps[li, h]:>9.3f}  "
                f"{np.nanmean(norm_stack, axis=0)[li, h]:>9.3f}  {mean_diff[li, h]:>+9.3f}  "
                f"{rho:>14.3f} (p={pval:.3f})"
            )
        return rows

    top_by_gps = _top(mean_gps, "attn_gps")
    top_by_diff = _top(mean_diff, "attn_gps − attn_norm")

    return {
        "n_pairs": len(nv),
        "pair_info": per_pair,
        "top_heads_by_attn_gps": top_by_gps,
        "top_heads_by_diff": top_by_diff,
        "mean_attn_gps": mean_gps.tolist(),
        "mean_attn_diff": mean_diff.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────
# Step 11 — activation patching NORMAL → GPS
# ──────────────────────────────────────────────────────────────────────────


class _CaptureLayers:
    """Forward hooks on every decoder layer, storing output hidden states."""

    def __init__(self, model):
        layers = decoder_layers(model)
        self.captured: list[torch.Tensor | None] = [None] * len(layers)
        self._handles = [
            layer.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(layers)
        ]

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.captured[idx] = hidden.detach().clone()
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


class _PatchLayerPositions:
    """Forward hook on one decoder layer: overwrite the output hidden state at
    the given positions with the given vectors (in place)."""

    def __init__(self, layer_module, positions: list[int], vectors: list[torch.Tensor]):
        self.positions = positions
        self.vectors = vectors
        self._handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        for pos, vec in zip(self.positions, self.vectors, strict=True):
            hidden[:, pos, :] = vec.to(hidden.dtype)

    def remove(self) -> None:
        self._handle.remove()


def _aligned_positions(
    tokenizer, src_text: str, tgt_text: str, word_char_spans: list[tuple[int, int, int, int]]
) -> list[tuple[list[int], list[int]]]:
    """Token indices of each shared word in src_text and tgt_text.

    `word_char_spans` holds (src_start, src_end, tgt_start, tgt_end) per word.
    Returns [(src_indices, tgt_indices), ...]; a word whose subword split
    differs between the two contexts is dropped."""
    out = []
    for ss, se, ts, te in word_char_spans:
        _, src_idx = _word_token_indices(tokenizer, src_text, ss, se)
        _, tgt_idx = _word_token_indices(tokenizer, tgt_text, ts, te)
        if src_idx and len(src_idx) == len(tgt_idx):
            out.append((src_idx, tgt_idx))
    return out


def run_patching_sweep(model, tokenizer, pairs: list[Pair], device: str) -> dict:
    """Denoising activation patching, per NV pair.

    Source run: NORMAL prefix + critical word (+ continuation) — the model
    parses the V-amb word as a verb without ambiguity. Target run: GPS
    prefix + critical word (+ continuation). We patch the source run's hidden
    state into the target run at the shared-surface positions (the V-amb word
    and the critical word), one layer at a time, and measure how much of the
    GPS→NORMAL comprehension gap is recovered:

        recovery(L, scope) = (s_gps − s_patched) / (s_gps − s_normal)

    A layer where patching the V-amb position recovers most of the gap is
    where the verb-vs-noun reading of the ambiguous word is causally carried.
    """
    nv = [p for p in pairs if p.template == "NV" and p.verb_continuation]
    n_layers, _, _ = get_architecture(model)
    scopes = ["v_amb", "critical", "both"]

    per_pair: list[dict] = []
    # accumulate s_patched per (layer, scope) across pairs
    acc = {sc: np.full((n_layers, len(nv)), np.nan) for sc in scopes}

    for pi, p in enumerate(nv):
        v_amb = _v_amb_word(p)
        src_prefix = p.normal_prefix + " " + p.critical_word
        tgt_prefix = p.gps_prefix + " " + p.critical_word
        src_text = src_prefix + " " + p.verb_continuation
        tgt_text = tgt_prefix + " " + p.verb_continuation

        s_gps = surprisal(model, tokenizer, tgt_prefix, p.verb_continuation, device)
        s_norm = surprisal(model, tokenizer, src_prefix, p.verb_continuation, device)

        # char spans of the shared words in both texts
        spans = [
            (  # V-amb word: last word of each prefix's NP part
                len(p.normal_prefix) - len(v_amb), len(p.normal_prefix),
                len(p.gps_prefix) - len(v_amb), len(p.gps_prefix),
            ),
            (  # critical word
                len(p.normal_prefix) + 1, len(p.normal_prefix) + 1 + len(p.critical_word),
                len(p.gps_prefix) + 1, len(p.gps_prefix) + 1 + len(p.critical_word),
            ),
        ]
        aligned = _aligned_positions(tokenizer, src_text, tgt_text, spans)
        if len(aligned) < 2:
            print(f"    pair {p.id}: token alignment failed, skipped", flush=True)
            continue
        scope_positions = {
            "v_amb": [aligned[0]],
            "critical": [aligned[1]],
            "both": aligned,
        }

        # capture the source run once
        cap = _CaptureLayers(model)
        try:
            src_ids = torch.tensor(
                [tokenizer(src_text, add_special_tokens=True)["input_ids"]], device=device
            )
            with torch.no_grad():
                model(src_ids)
            captured = [c for c in cap.captured]
        finally:
            cap.remove()

        for li in range(n_layers):
            for sc in scopes:
                positions, vectors = [], []
                for src_idx, tgt_idx in scope_positions[sc]:
                    for s_i, t_i in zip(src_idx, tgt_idx, strict=True):
                        positions.append(t_i)
                        vectors.append(captured[li][0, s_i])
                patch = _PatchLayerPositions(decoder_layers(model)[li], positions, vectors)
                try:
                    s_patched = surprisal(
                        model, tokenizer, tgt_prefix, p.verb_continuation, device
                    )
                finally:
                    patch.remove()
                acc[sc][li, pi] = s_patched

        per_pair.append({"id": p.id, "v_amb": v_amb, "s_gps": s_gps, "s_norm": s_norm})
        print(
            f"    pair {p.id:>3} ({v_amb}): s_gps={s_gps:.3f} s_norm={s_norm:.3f} done",
            flush=True,
        )

    mean_gps = float(np.mean([r["s_gps"] for r in per_pair]))
    mean_norm = float(np.mean([r["s_norm"] for r in per_pair]))
    gap = mean_gps - mean_norm

    print(f"\n  Baselines: mean s_gps = {mean_gps:.3f}, mean s_norm = {mean_norm:.3f}, gap = {gap:+.3f}")
    print("\n  Mean recovery of the GPS→NORMAL gap per layer (patched positions):")
    print(f"    {'layer':>5}  {'v_amb':>8}  {'critical':>8}  {'both':>8}  bar(both)")
    print("    " + "-" * 60)
    per_layer = []
    for li in range(n_layers):
        row = {"layer": li}
        for sc in scopes:
            m = float(np.nanmean(acc[sc][li]))
            row[f"s_patched_{sc}"] = m
            row[f"recovery_{sc}"] = float((mean_gps - m) / gap) if gap else float("nan")
        per_layer.append(row)
        rec = row["recovery_both"]
        bar = "" if np.isnan(rec) else "█" * max(0, min(40, int(rec * 40)))
        print(
            f"    {li:>5}  {row['recovery_v_amb']:>+8.3f}  {row['recovery_critical']:>+8.3f}  "
            f"{row['recovery_both']:>+8.3f}  {bar}",
            flush=True,
        )

    best = max(per_layer, key=lambda r: r["recovery_both"])
    print(
        f"\n  Best layer: L{best['layer']} recovers {best['recovery_both']:+.1%} of the gap "
        f"(v_amb only: {best['recovery_v_amb']:+.1%})"
    )

    return {
        "n_pairs": len(per_pair),
        "mean_s_gps": mean_gps,
        "mean_s_norm": mean_norm,
        "per_pair_baselines": per_pair,
        "per_layer": per_layer,
    }


# ──────────────────────────────────────────────────────────────────────────
# Step 12 — mean ablation
# ──────────────────────────────────────────────────────────────────────────


def collect_head_means(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    """Mean o_proj input per layer over all positions of `texts`.

    Returns tensor (n_layers, n_heads * head_dim) on CPU, float32."""
    n_layers, n_heads, head_dim = get_architecture(model)
    dim = n_heads * head_dim
    sums = torch.zeros(n_layers, dim, dtype=torch.float64)
    counts = torch.zeros(n_layers, dtype=torch.float64)

    handles = []

    def make_hook(idx: int):
        def hook(_module, args):
            x = args[0]  # (B, T, dim)
            # MPS has no float64: accumulate in fp32 on device, widen on CPU
            sums[idx] += x.detach().reshape(-1, dim).float().sum(dim=0).cpu().double()
            counts[idx] += x.shape[0] * x.shape[1]
        return hook

    for i, layer in enumerate(decoder_layers(model)):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(i)))
    try:
        for text in texts:
            ids = torch.tensor(
                [tokenizer(text, add_special_tokens=True)["input_ids"]], device=device
            )
            with torch.no_grad():
                model(ids)
    finally:
        for h in handles:
            h.remove()

    return (sums / counts.unsqueeze(1)).float()


class HeadsMeanAblated:
    """Mean-ablation twin of HeadsAblated: while active, each target head's
    o_proj input slice is replaced with its dataset mean instead of zero.

    Zero-ablation pushes activations off-distribution; replacing with the
    mean keeps the layer's input statistics intact and is the recommended
    practice for causal claims (Zhang & Nanda 2024)."""

    def __init__(self, model, head_dim: int, targets, means: torch.Tensor):
        self.model = model
        self.head_dim = head_dim
        self.means = means
        by_layer: dict[int, list[int]] = {}
        for li, h in targets:
            by_layer.setdefault(li, []).append(h)
        self._by_layer = by_layer
        self._handles = []

    def __enter__(self) -> HeadsMeanAblated:
        for li, heads in self._by_layer.items():
            o_proj = decoder_layers(self.model)[li].self_attn.o_proj
            mean_row = self.means[li]

            def make_hook(heads=heads, mean_row=mean_row):
                def hook(_module, args):
                    x = args[0]
                    for h in heads:
                        s, e = h * self.head_dim, (h + 1) * self.head_dim
                        x[..., s:e] = mean_row[s:e].to(x.device, x.dtype)
                return hook

            self._handles.append(o_proj.register_forward_pre_hook(make_hook()))
        return self

    def __exit__(self, *_exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def mean_ablation_texts(pairs: list[Pair]) -> list[str]:
    """The distribution the means are computed over: every GPS and NORMAL
    sentence plus the control probes."""
    from _ablation import CONTROL_PROBES

    texts = []
    for p in pairs:
        texts.append(p.gps)
        texts.append(p.normal)
    texts += [prefix + " " + word for prefix, word in CONTROL_PROBES]
    return texts


def run_mean_ablation(
    model,
    tokenizer,
    device: str,
    head_sets: list[list[tuple[int, int]]],
    layers: list[int],
) -> dict:
    """Replicate the key zero-ablation results with mean ablation.

    For each head set: Δ-surprisal effect (steps 02-03 claim) and verb
    continuation (step 04 claim). For each layer: all-heads mean ablation
    with the comprehension metric (step 05 claim)."""
    n_layers, n_heads, head_dim = get_architecture(model)

    print("  computing per-head means over the pair+control distribution ...", flush=True)
    means = collect_head_means(model, tokenizer, mean_ablation_texts(PAIRS), device)

    print("  baselines (no ablation) ...", flush=True)
    base_eff, _ = measure_effect(model, tokenizer, PAIRS, device)
    base_vc, _ = measure_verb_continuation(model, tokenizer, PAIRS, device)

    def _report(label: str, targets: list[tuple[int, int]]) -> dict:
        with HeadsAblated(model, head_dim, targets):
            zero_eff, _ = measure_effect(model, tokenizer, PAIRS, device)
            zero_vc, _ = measure_verb_continuation(model, tokenizer, PAIRS, device)
        with HeadsMeanAblated(model, head_dim, targets, means):
            mean_eff, _ = measure_effect(model, tokenizer, PAIRS, device)
            mean_vc, _ = measure_verb_continuation(model, tokenizer, PAIRS, device)
        row = {
            "label": label,
            "targets": [list(t) for t in targets],
            "zero": {
                "effect_drop": zero_eff["mean_delta"] - base_eff["mean_delta"],
                "vc_gps_inflation": zero_vc["mean_s_vc_gps"] - base_vc["mean_s_vc_gps"],
                "control_inflation": zero_vc["mean_s_control"] - base_vc["mean_s_control"],
            },
            "mean": {
                "effect_drop": mean_eff["mean_delta"] - base_eff["mean_delta"],
                "vc_gps_inflation": mean_vc["mean_s_vc_gps"] - base_vc["mean_s_vc_gps"],
                "control_inflation": mean_vc["mean_s_control"] - base_vc["mean_s_control"],
            },
        }
        print(
            f"    {label:<16}  Δ-drop  zero={row['zero']['effect_drop']:+7.3f}  mean={row['mean']['effect_drop']:+7.3f}   "
            f"+vc_gps  zero={row['zero']['vc_gps_inflation']:+7.3f}  mean={row['mean']['vc_gps_inflation']:+7.3f}   "
            f"+ctrl  zero={row['zero']['control_inflation']:+7.3f}  mean={row['mean']['control_inflation']:+7.3f}",
            flush=True,
        )
        return row

    print("\n  Head sets (zero vs mean ablation):")
    head_rows = [
        _report("+".join(f"L{li}.H{h}" for li, h in ts), ts) for ts in head_sets
    ]
    print("\n  Full layers (zero vs mean ablation):")
    layer_rows = [
        _report(f"L{li} (all heads)", [(li, h) for h in range(n_heads)]) for li in layers
    ]

    return {
        "baseline_effect": base_eff,
        "baseline_vc": base_vc,
        "head_sets": head_rows,
        "layers": layer_rows,
    }


# ──────────────────────────────────────────────────────────────────────────
# Step 13 — random single-head ablation null distribution
# ──────────────────────────────────────────────────────────────────────────


def run_null_distribution(
    model, tokenizer, device: str, n_samples: int = 60, seed: int = 42
) -> dict:
    """Empirical null for single-head zero-ablation effects.

    Samples random (layer, head) pairs and measures, for each, the same
    statistics the project's claims are stated in:
      - NV Δ-surprisal effect drop        (step 03: "L25.H7 drops −0.72 on NV")
      - vc_gps inflation and selectivity  (step 05: "+0.25 is noise")

    The percentiles of these distributions are what "noise" means."""
    n_layers, n_heads, head_dim = get_architecture(model)
    nv_pairs = by_template("NV")

    rng = np.random.default_rng(seed)
    all_combos = [(li, h) for li in range(n_layers) for h in range(n_heads)]
    idx = rng.choice(len(all_combos), size=min(n_samples, len(all_combos)), replace=False)
    samples = [all_combos[i] for i in idx]

    print("  baselines (no ablation, NV pairs) ...", flush=True)
    base_eff, _ = measure_effect(model, tokenizer, nv_pairs, device)
    base_vc, _ = measure_verb_continuation(model, tokenizer, nv_pairs, device)

    rows = []
    print(f"  sampling {len(samples)} random heads ...")
    print(f"    {'head':>8}  {'nv_drop':>8}  {'+vc_gps':>8}  {'+ctrl':>8}  {'select':>8}")
    print("    " + "-" * 48)
    for li, h in samples:
        with HeadsAblated(model, head_dim, [(li, h)]):
            eff, _ = measure_effect(model, tokenizer, nv_pairs, device)
            vc, _ = measure_verb_continuation(model, tokenizer, nv_pairs, device)
        row = {
            "layer": int(li),
            "head": int(h),
            "nv_effect_drop": eff["mean_delta"] - base_eff["mean_delta"],
            "vc_gps_inflation": vc["mean_s_vc_gps"] - base_vc["mean_s_vc_gps"],
            "control_inflation": vc["mean_s_control"] - base_vc["mean_s_control"],
        }
        row["selectivity"] = row["vc_gps_inflation"] - row["control_inflation"]
        rows.append(row)
        print(
            f"    L{li:>2}.H{h:<3}  {row['nv_effect_drop']:>+8.3f}  "
            f"{row['vc_gps_inflation']:>+8.3f}  {row['control_inflation']:>+8.3f}  "
            f"{row['selectivity']:>+8.3f}",
            flush=True,
        )

    def _percentiles(key: str) -> dict:
        a = np.array([r[key] for r in rows])
        return {
            "p5": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "abs_p90": float(np.percentile(np.abs(a), 90)),
            "abs_p95": float(np.percentile(np.abs(a), 95)),
            "abs_max": float(np.abs(a).max()),
        }

    summary = {k: _percentiles(k) for k in ("nv_effect_drop", "vc_gps_inflation", "selectivity")}
    print("\n  Null percentiles:")
    for k, v in summary.items():
        print(
            f"    {k:<18}  p5={v['p5']:+.3f}  p50={v['p50']:+.3f}  p95={v['p95']:+.3f}  "
            f"|.|p95={v['abs_p95']:.3f}  |.|max={v['abs_max']:.3f}"
        )

    return {"n_samples": len(rows), "samples": rows, "percentiles": summary}


def run_null_distribution_vc(
    model, tokenizer, device: str, n_samples: int = 300, seed: int = 43
) -> dict:
    """Empirical null for single-head ablation on the comprehension metric.

    The step-06 claim is a *selected maximum*: "the best of the K heads swept
    inside the step-05 layers inflates vc_gps by X". The fair threshold is
    therefore the null distribution of the max over K random heads, not the
    single-head percentile. The vc-only probe is cheap (10 NV pairs + the
    controls), so this samples many more heads than step 13 and bootstraps
    max-of-K thresholds for K = n_heads and K = 2*n_heads (one and two swept
    layers, the TOP_VCLAYERS default)."""
    n_layers, n_heads, head_dim = get_architecture(model)
    nv_pairs = by_template("NV")

    rng = np.random.default_rng(seed)
    all_combos = [(li, h) for li in range(n_layers) for h in range(n_heads)]
    idx = rng.choice(len(all_combos), size=min(n_samples, len(all_combos)), replace=False)
    samples = [all_combos[i] for i in idx]

    print("  baseline (no ablation, NV pairs) ...", flush=True)
    base_vc, _ = measure_verb_continuation(model, tokenizer, nv_pairs, device)

    rows = []
    print(f"  sampling {len(samples)} random heads (vc metric only) ...")
    print(f"    {'head':>8}  {'+vc_gps':>8}  {'+vc_delta':>9}  {'+ctrl':>8}  {'select':>8}")
    print("    " + "-" * 50)
    for li, h in samples:
        with HeadsAblated(model, head_dim, [(li, h)]):
            vc, _ = measure_verb_continuation(model, tokenizer, nv_pairs, device)
        row = {
            "layer": int(li),
            "head": int(h),
            "vc_gps_inflation": vc["mean_s_vc_gps"] - base_vc["mean_s_vc_gps"],
            "vc_delta_inflation": vc["mean_vc_delta"] - base_vc["mean_vc_delta"],
            "control_inflation": vc["mean_s_control"] - base_vc["mean_s_control"],
        }
        row["selectivity"] = row["vc_gps_inflation"] - row["control_inflation"]
        rows.append(row)
        print(
            f"    L{li:>2}.H{h:<3}  {row['vc_gps_inflation']:>+8.3f}  "
            f"{row['vc_delta_inflation']:>+9.3f}  {row['control_inflation']:>+8.3f}  "
            f"{row['selectivity']:>+8.3f}",
            flush=True,
        )

    def _percentiles(key: str) -> dict:
        a = np.array([r[key] for r in rows])
        return {
            "p5": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "abs_p95": float(np.percentile(np.abs(a), 95)),
            "abs_max": float(np.abs(a).max()),
        }

    summary = {
        k: _percentiles(k)
        for k in ("vc_gps_inflation", "vc_delta_inflation", "selectivity")
    }

    # Selection-corrected thresholds: distribution of max(vc_gps_inflation)
    # over K heads drawn from the null, bootstrapped from the samples above.
    boot_rng = np.random.default_rng(seed + 1)
    a = np.array([r["vc_gps_inflation"] for r in rows])
    max_of_k = {}
    for k in (n_heads, 2 * n_heads):
        draws = boot_rng.choice(a, size=(10_000, k), replace=True).max(axis=1)
        max_of_k[str(k)] = {
            "p50": float(np.percentile(draws, 50)),
            "p90": float(np.percentile(draws, 90)),
            "p95": float(np.percentile(draws, 95)),
            "p99": float(np.percentile(draws, 99)),
        }

    print("\n  Null percentiles (single head):")
    for k, v in summary.items():
        print(
            f"    {k:<20}  p5={v['p5']:+.3f}  p50={v['p50']:+.3f}  p95={v['p95']:+.3f}  "
            f"p99={v['p99']:+.3f}  |.|max={v['abs_max']:.3f}"
        )
    print("\n  Selection-corrected: null max(vc_gps_inflation) over K random heads:")
    for k, v in max_of_k.items():
        print(
            f"    K={k:>3}  p50={v['p50']:+.3f}  p90={v['p90']:+.3f}  "
            f"p95={v['p95']:+.3f}  p99={v['p99']:+.3f}"
        )

    return {
        "n_samples": len(rows),
        "samples": rows,
        "percentiles": summary,
        "max_of_k": max_of_k,
    }
