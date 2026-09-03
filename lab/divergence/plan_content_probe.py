"""Is the plan EMPTY, or FULL BUT UNREAD? A blind decoder reads z directly.

The TUL campaign has circled this question for weeks: every measurement so far reads
the plan THROUGH the coda (plan-off ablations in `slot_path_worth.py`), which cannot
tell "the plan carries nothing" apart from "the plan carries plenty and the coda just
never asks for it" — both give the same near-zero plan-off cost. This probe sidesteps
the coda entirely: freeze the checkpoint, extract z, and train a SMALL, BLIND, throwaway
decoder to read span content straight off z. `CLAUDE.md`'s ban on decoding a span from
one vector with no token path is about the SHIPPED path (Huginn/MegaByte/Bowman/
Hourglass); a frozen-weights measurement with a decoder that is never shipped does not
touch that ban — see the spec's "Why this shape" section.

Second, free once the harness exists: is z a SUMMARY of its own span, or a PLAN for the
next one? TUL's claim is the latter.

**What z is.** The output of `TULSlots.prefix_project` — what the coda actually reads —
NOT the raw `h_slots` looped carrier. Reading the carrier would measure a channel the
coda never sees and would overstate the plan (spec, "What z is").

**The four conditions**, one decoder class, identical (row, slot) example set, identical
size/steps/LR/seed:

| condition | z fed to decoder            | target   | measures                          |
|-----------|------------------------------|----------|------------------------------------|
| PLAN      | z_i (real)                   | span i+1 | the TUL claim                      |
| SUMMARY   | z_i (real)                   | span i   | is z a summary instead?            |
| SHUFFLED  | z from a random OTHER row    | span i+1 | THE DECIDING CONTROL               |
| POSITION  | zeros (offset only)          | span i+1 | what offset alone predicts         |

SHUFFLED holds the decoder, the target distribution, and the offset structure fixed and
destroys ONLY the z-to-span correspondence — across-row (not within-row, which can leak
topic between two slots of the same document).

Decision rule (pre-registered in `probe-spec.md`, restated here so the printed report is
self-contained). NOTE on sign: nats/token is a LOSS (lower = better); the spec's own
decision bands (positive number = informative) only hold for (worse condition's nats)
MINUS (better condition's nats), so that is what is computed and printed under the
spec's own labels — see the "SIGN NOTE" comment at the print site in `main()`:
    PLAN - SHUFFLED >= 0.20 nats/token  -> FULL   (coda not reading it; fix the reader)
    PLAN - SHUFFLED <= 0.05 nats/token  -> EMPTY  (fix provenance/objective, not the reader)
    otherwise                            -> INCONCLUSIVE
    SUMMARY - PLAN   >= 0.20 nats/token -> z looks like a summary, not a plan (headline
                                            finding regardless of the PLAN number)

Usage:
    python lab/divergence/plan_content_probe.py \\
        --ckpt checkpoints/morph/tg2-s1/step_3000.pt --config tul_tg2 \\
        --fit-batches 6 --eval-batches 2 --out /tmp/tg2-s1.json
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.model.attention import RMSNorm                   # noqa: E402
from morph.model.tul_layout import SlotLayout                # noqa: E402
from morph.training.data import create_dataloader            # noqa: E402
from morph.training.train import load_checkpoint             # noqa: E402

__all__ = [
    "BlindSpanDecoder", "reduce_prefix_values", "extract_slot_examples",
    "shuffled_other_row", "fit_decoder", "eval_decoder", "unigram_floor",
    "run_four_conditions", "capture_prefix_project", "assert_frozen",
    "param_fingerprint", "assert_disjoint_batches",
]


# ── z extraction ──────────────────────────────────────────────────────────────

@contextlib.contextmanager
def capture_prefix_project(root: nn.Module, sink: list[Tensor]):
    """Intercept `TULSlots.prefix_project`'s OUTPUT — the exact values the coda reads —
    by monkey-patching the bound method. This is the SAME technique `plan_off` in
    `slot_path_worth.py` already uses at this call site (there to rewrite the value;
    here only to record it, transparently — the real forward runs unmodified).

    NOT a `nn.Module.register_forward_hook`: `prefix_project` is a plain method on
    `TULSlots`, not itself a submodule with its own `forward`, so there is nothing for
    `register_forward_hook` to attach to. The wrapper below is a passthrough (it returns
    exactly what `orig` returned), so every downstream branch of `_forward_tul` — TG
    restriction, the gate, sigreg, mux, the coda itself — runs exactly as it would
    outside a probe. This is eval-only with no checkpointing in play, so unlike a
    training-time side channel (the `ret_capture` lesson) this capture is safe: nothing
    here is saved to or restored from a checkpoint.
    """
    tul = root.tul
    orig = tul.prefix_project

    def wrapped(h_slots, layout, l_total):
        values, pos = orig(h_slots, layout, l_total)
        sink.append(values.detach().clone())
        return values, pos

    tul.prefix_project = wrapped
    try:
        yield
    finally:
        tul.prefix_project = orig


def reduce_prefix_values(values: Tensor, S: int, K: int) -> Tensor:
    """``[B, S*K, *mid, C]`` (prefix_project's raw output) -> ``[B, S, K*C]``, one flat
    z vector per slot. ``mid`` is ``()`` for a plain carrier or ``(n,)`` for the Cayley
    HyperConnection carrier (n=hc_streams, default 4).

    Two DIFFERENT reductions for two DIFFERENT reasons:

    1. ``n -> mean`` (when ``mid`` is non-empty). This matches the model's OWN reduction:
       ``MORPHTransformer._readout`` collapses the same HC n-stream carrier with
       ``x.mean(dim=2)`` right before the LM head ("Mean readout is scale-preserving …
       exactly recovers the plain-residual output" — transformer.py). Any other
       reduction here would hand the probe a channel the coda's own readout never uses.
    2. ``K -> concat``, never mean. The ``prefix_k`` positions are NOT interchangeable:
       spec §3.1 states the first ``K-1`` carry "the plan with NO label" and the LAST one
       predicts the first token of the next span (``pack_tul_row``'s ``emit_pos``).
       Averaging them would blur a distinction the model itself preserves; concatenating
       keeps every position's content and still yields one FIXED-size vector per slot
       (``prefix_k`` is a construction-time constant, identical for every slot).
    """
    B = values.shape[0]
    C = values.shape[-1]
    mid = values.shape[2:-1]
    v = values.reshape(B, S, K, *mid, C)
    if mid:
        v = v.mean(dim=tuple(range(3, 3 + len(mid))))          # -> [B, S, K, C]
    return v.reshape(B, S, K * C)


def _pad_target(tokens: Tensor, span_cap: int) -> Tensor:
    out = torch.full((span_cap,), -100, dtype=torch.long)
    out[: tokens.numel()] = tokens.to(torch.long)
    return out


@dataclass
class SlotExamples:
    """One row per usable (row, slot) pair. All CPU tensors.

    ``z``:        ``[N, z_dim]`` — the slot's reduced `prefix_project` output.
    ``span_i``:   ``[N, span_cap]`` int64 — SUMMARY target (this slot's OWN span),
                  -100 padding past the true length.
    ``span_ip1``: ``[N, span_cap]`` int64 — PLAN target (the NEXT span), -100 padding.
    ``row_id``:   ``[N]`` int64 — globally unique row id (batch offset + row-in-batch),
                  used to build the across-row SHUFFLED control.
    ``n_excluded``: valid slots dropped because they are the LAST valid slot of their
                  row (spec: "the last valid slot in a row has no span i+1" — the
                  packer's open tail sits in the dump bin, not a real `bag_id`).
    ``n_total_valid``: valid (non-pad) slots seen, before that exclusion.
    """

    z: Tensor
    span_i: Tensor
    span_ip1: Tensor
    row_id: Tensor
    n_excluded: int
    n_total_valid: int


def extract_slot_examples(input_ids: Tensor, z: Tensor, layout: SlotLayout,
                          span_cap: int, row_offset: int) -> SlotExamples:
    """Pull one training example per slot that has BOTH its own span (SUMMARY target)
    and a next span (PLAN target). ``z`` is ``[B, S, z_dim]`` (already reduced by
    :func:`reduce_prefix_values`); ``input_ids`` is ``[B, L]``, the actual tokens.

    Span membership follows `pack_tul_row`/`mux_span_targets`'s convention exactly: a
    TOKEN position's `bag_id` is the index of the slot that CLOSES its span, so span
    `s`'s tokens are `(bag_id == s) & ~slot_mask`. Slot `s` is built from span `s` (its
    OWN summary) and sits immediately before span `s+1` in the row, which is what the
    coda actually conditions on next — hence PLAN's target is `bag_id == s+1`.
    """
    B, S, zdim = z.shape
    bag_id, slot_mask, slot_valid = layout.bag_id, layout.slot_mask, layout.slot_valid
    out_z, out_i, out_ip1, out_rid = [], [], [], []
    n_total_valid, n_excluded = 0, 0
    for b in range(B):
        n_valid = int(slot_valid[b].sum().item())
        n_total_valid += n_valid
        if n_valid < 2:
            n_excluded += n_valid    # 0 or 1 valid slots: none of them are usable
            continue
        n_excluded += 1              # the last valid slot of this row has no span s+1
        bag_b, tok_b, ids_b = bag_id[b], ~slot_mask[b], input_ids[b]
        for s in range(n_valid - 1):
            idx_i = torch.nonzero((bag_b == s) & tok_b, as_tuple=True)[0]
            idx_ip1 = torch.nonzero((bag_b == s + 1) & tok_b, as_tuple=True)[0]
            if idx_i.numel() == 0 or idx_ip1.numel() == 0:
                # min_span >= 4 (docs/tul-spec.md) makes an empty span for a slot INSIDE
                # n_valid - 1 impossible by construction. An empty span here means this
                # function's bag_id/slot_valid bookkeeping disagrees with the packer's —
                # a bug, not a data edge case — so it raises rather than silently
                # producing an empty target.
                raise RuntimeError(
                    f"row {b} slot {s}: empty span (span_i={idx_i.numel()} tokens, "
                    f"span_i+1={idx_ip1.numel()} tokens) — extraction/layout mismatch")
            span_i_tok, span_ip1_tok = ids_b[idx_i], ids_b[idx_ip1]
            if span_i_tok.numel() > span_cap or span_ip1_tok.numel() > span_cap:
                raise RuntimeError(
                    f"row {b} slot {s}: span longer than span_cap={span_cap} "
                    f"({span_i_tok.numel()}/{span_ip1_tok.numel()} tokens) — the "
                    f"boundary rule is supposed to force a cut at span_cap")
            out_z.append(z[b, s].detach().cpu())
            out_i.append(_pad_target(span_i_tok.cpu(), span_cap))
            out_ip1.append(_pad_target(span_ip1_tok.cpu(), span_cap))
            out_rid.append(row_offset + b)
    z_out = torch.stack(out_z) if out_z else z.new_zeros(0, zdim).cpu()
    i_out = (torch.stack(out_i) if out_i
             else torch.zeros(0, span_cap, dtype=torch.long))
    ip1_out = (torch.stack(out_ip1) if out_ip1
               else torch.zeros(0, span_cap, dtype=torch.long))
    rid_out = torch.tensor(out_rid, dtype=torch.long)
    return SlotExamples(z_out, i_out, ip1_out, rid_out, n_excluded, n_total_valid)


def cat_examples(parts: list[SlotExamples]) -> SlotExamples:
    return SlotExamples(
        z=torch.cat([p.z for p in parts]),
        span_i=torch.cat([p.span_i for p in parts]),
        span_ip1=torch.cat([p.span_ip1 for p in parts]),
        row_id=torch.cat([p.row_id for p in parts]),
        n_excluded=sum(p.n_excluded for p in parts),
        n_total_valid=sum(p.n_total_valid for p in parts),
    )


# ── the SHUFFLED control ────────────────────────────────────────────────────

def shuffled_other_row(z: Tensor, row_id: Tensor, seed: int) -> Tensor:
    """Permute ``z`` ACROSS ROWS: example ``i``'s replacement is drawn from a uniformly
    random OTHER row, then a uniformly random slot within that row. Across-row rather
    than within-row per the spec's guard: two slots of the SAME document/row can share
    topic, which a within-row shuffle would leak back in and understate how much the
    control destroys.
    """
    rows = sorted(set(row_id.tolist()))
    if len(rows) < 2:
        raise RuntimeError(
            "need >= 2 distinct rows to build an across-row SHUFFLED control "
            f"(got {len(rows)})")
    row_to_idx = {r: torch.nonzero(row_id == r, as_tuple=True)[0] for r in rows}
    rng = random.Random(seed)
    picks = []
    for r in row_id.tolist():
        other = r
        while other == r:
            other = rng.choice(rows)
        pool = row_to_idx[other]
        picks.append(int(pool[rng.randrange(len(pool))]))
    return z[torch.tensor(picks, dtype=torch.long)]


# ── the decoder ───────────────────────────────────────────────────────────────

class BlindSpanDecoder(nn.Module):
    """``z -> MLP -> K x d -> FRESH unembedding -> logits[K, V]``. Blind: no token input
    at any position, ever — the offsets are read off a learned embedding table indexed
    by position alone, never by any token this decoder is trying to predict.

    **Weight-tying hazard (memory: morph-lm-head-is-weight-tied).** ``lm_weight()`` IS
    MORPH's input embedding table. ``unembed`` below is a FRESH ``nn.Linear`` that has
    never touched any MORPH tensor — not a ``.detach()``ed alias of one, a fully
    independent parameter — so this decoder cannot train MORPH's embeddings even in
    principle, and there is nothing to detach.

    ``POSITION`` condition: callers pass ``z = torch.zeros_like(real_z)`` rather than a
    smaller network. Same class, same parameter count, same forward path, in every
    condition — only the INPUT changes — which is what makes the four conditions'
    "identical decoder size" comparable by construction rather than by later checking
    that two different classes happened to end up the same size.
    """

    def __init__(self, z_dim: int, hidden: int, k: int, vocab_size: int):
        super().__init__()
        self.k = k
        self.in_proj = nn.Linear(z_dim, hidden)
        self.offset_embed = nn.Embedding(k, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))
        self.norm = RMSNorm(hidden)
        self.unembed = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        base = self.in_proj(z).unsqueeze(1)                      # [N, 1, hidden]
        h = base + self.offset_embed.weight.unsqueeze(0)          # [N, K, hidden]
        h = h + self.mlp(h)
        h = self.norm(h)
        return self.unembed(h)                                    # [N, K, V]


def fit_decoder(z: Tensor, targets: Tensor, *, vocab_size: int, hidden: int, steps: int,
                lr: float, batch_size: int, seed: int, device: torch.device,
                weight_decay: float = 1e-2) -> tuple[BlindSpanDecoder, float]:
    """Train ONE fresh :class:`BlindSpanDecoder` on ``(z, targets)``.

    ``targets`` ``[N, K]`` int64, -100 (torch's `ignore_index` convention) at offsets
    past the true span length. Mini-batched (random-with-replacement draws): a full-batch
    pass would materialise ``[N, K, V]`` logits at once, which at real-checkpoint scale
    (N in the thousands, K=span_cap up to 32, V~49k) is tens of GB — mini-batching is the
    memory guard, not a training-quality choice. Returns ``(decoder, final train loss)``.

    ``weight_decay`` is NOT a tuning knob, it is a correctness guard, found by the
    self-test's negative control: with z continuous and independent of the target, and
    with N in the low hundreds (or z_dim large relative to N), an un-regularised decoder
    can MEMORISE individual fit-set z vectors — training loss drops far below the
    marginal entropy, and the eval loss on FRESH (uncorrelated) z then comes out WORSE
    than a decoder that never looked at z at all, wrongly reading as "informative" in the
    wrong direction. `weight_decay=1e-2` was the smallest value on a synthetic sweep
    (0, 1e-3, 1e-2, 5e-2) that pushed PLAN-vs-SHUFFLED back to ~0 on pure noise while
    leaving a real signal (the positive control) essentially untouched — see
    `tests/test_plan_content_probe.py`'s controls.
    """
    torch.manual_seed(seed)
    dec = BlindSpanDecoder(z.shape[-1], hidden, targets.shape[-1], vocab_size).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=lr, weight_decay=weight_decay)
    n = z.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    last_loss = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=g)
        zb, tb = z[idx].to(device), targets[idx].to(device)
        logits = dec(zb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), tb.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
    return dec, last_loss


@torch.no_grad()
def eval_decoder(dec: BlindSpanDecoder, z: Tensor, targets: Tensor, *, vocab_size: int,
                 batch_size: int, device: torch.device) -> tuple[float, int]:
    """Nats/token on held-out ``(z, targets)``, mini-batched for the same memory reason
    as :func:`fit_decoder`. Returns ``(nats_per_token, n_scored_tokens)``."""
    dec.eval()
    n = z.shape[0]
    total_nll, total_n = 0.0, 0
    for start in range(0, n, batch_size):
        zb = z[start:start + batch_size].to(device)
        tb = targets[start:start + batch_size].to(device)
        logits = dec(zb)
        n_valid = int((tb != -100).sum())
        if n_valid == 0:
            continue
        nll = F.cross_entropy(logits.reshape(-1, vocab_size), tb.reshape(-1),
                              ignore_index=-100, reduction="sum")
        total_nll += float(nll)
        total_n += n_valid
    dec.train()
    return total_nll / max(total_n, 1), total_n


# ── memorization gate ─────────────────────────────────────────────────────────

CANARY_MAX = 0.5


def memorization_gate(position: dict, max_gap: float = CANARY_MAX) -> tuple[float, bool]:
    """``(gap, readable)`` from the POSITION condition's train/eval split.

    `fit_decoder`'s docstring names the failure this guards: too little weight decay lets
    the decoder MEMORIZE the fit set, its train loss collapses below the marginal entropy,
    and eval loss on fresh z comes out WORSE than SHUFFLED — flipping the SIGN of the
    deciding number. `weight_decay=1e-2` was found on a SYNTHETIC sweep; nothing yet shows
    it holds at 49k vocab on a real checkpoint.

    POSITION is the canary, and it is free: it is handed NO z, so it cannot legitimately
    learn anything z-specific, while sharing every other condition's decoder size, step
    budget, LR and seed. Its train/eval gap is therefore memorization capacity that ALL
    four conditions had to spend. Above ``max_gap`` the panel is refused rather than
    reported — the `score_arms.py` convention, and the reason a too-coarse probe cadence
    there returns None instead of a noisier verdict.

    Added 2026-08-28, BEFORE any probe produced data, so the threshold cannot be fitted to
    a result. Recorded as a method amendment in
    ../experiments/planned/2026-08-28-plan-content.md.
    """
    gap = float(position["nats_per_token"]) - float(position["final_train_loss"])
    return gap, gap <= max_gap


# ── unigram floor ─────────────────────────────────────────────────────────────

def unigram_floor(fit_tokens: Tensor, eval_targets: Tensor, vocab_size: int
                  ) -> tuple[float, int]:
    """Corpus-unigram NLL floor. Counts come from ``fit_tokens`` (Laplace/add-1
    smoothed) — the SAME fit/eval split the decoder uses, never the eval set's own
    frequencies, which would be circular (an unigram model "fit" on the numbers it is
    then scored against is not a floor, it is a tautology)."""
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    flat = fit_tokens.reshape(-1)
    flat = flat[flat >= 0]
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
    logp = torch.log((counts + 1.0) / (counts.sum() + vocab_size))
    tgt = eval_targets.reshape(-1)
    valid = tgt != -100
    n = int(valid.sum())
    nll = -logp[tgt[valid]].sum().item()
    return nll / max(n, 1), n


# ── orchestration ─────────────────────────────────────────────────────────────

def run_four_conditions(fit: SlotExamples, ev: SlotExamples, *, vocab_size: int,
                        hidden: int, steps: int, lr: float, batch_size: int, seed: int,
                        device: torch.device, weight_decay: float = 1e-2) -> dict:
    """Fit + eval PLAN / SUMMARY / SHUFFLED / POSITION on the SAME (row, slot) example
    set (`fit`/`ev` already share `extract_slot_examples`'s exclusion of each row's last
    valid slot for ALL FOUR conditions, including SUMMARY — a stronger "identical budget"
    than matching decoder size/steps/seed alone, since it also equalises N)."""
    z_fit_shuf = shuffled_other_row(fit.z, fit.row_id, seed)
    z_eval_shuf = shuffled_other_row(ev.z, ev.row_id, seed + 1)

    conditions = {
        "PLAN":     (fit.z,                    fit.span_ip1, ev.z,                   ev.span_ip1),
        "SUMMARY":  (fit.z,                    fit.span_i,   ev.z,                   ev.span_i),
        "SHUFFLED": (z_fit_shuf,                fit.span_ip1, z_eval_shuf,            ev.span_ip1),
        "POSITION": (torch.zeros_like(fit.z),  fit.span_ip1, torch.zeros_like(ev.z), ev.span_ip1),
    }

    results, param_counts = {}, set()
    for name, (zf, tf, ze, te) in conditions.items():
        dec, train_loss = fit_decoder(zf, tf, vocab_size=vocab_size, hidden=hidden,
                                      steps=steps, lr=lr, batch_size=batch_size,
                                      seed=seed, device=device, weight_decay=weight_decay)
        n_params = sum(p.numel() for p in dec.parameters())
        param_counts.add(n_params)
        nats, n_tok = eval_decoder(dec, ze, te, vocab_size=vocab_size,
                                   batch_size=batch_size, device=device)
        n_masked = int(te.numel()) - n_tok
        results[name] = {
            "nats_per_token": nats, "n_eval_tokens": n_tok,
            "n_eval_masked_offsets": n_masked, "n_decoder_params": n_params,
            "final_train_loss": train_loss,
        }
    assert len(param_counts) == 1, (
        f"decoder parameter count differs across conditions: {param_counts} — POSITION "
        f"must keep the SAME architecture fed zeros, not a smaller network, or the "
        f"comparison is not apples-to-apples (spec: 'identical decoder size')")
    return results


# ── freeze guards ─────────────────────────────────────────────────────────────

def assert_frozen(model: nn.Module) -> None:
    bad = [n for n, p in model.named_parameters() if p.requires_grad]
    if bad:
        raise RuntimeError(
            f"{len(bad)} MORPH parameter(s) still require grad after "
            f"requires_grad_(False): {bad[:5]}{'...' if len(bad) > 5 else ''}. The "
            f"probe decoder must never be able to backprop into the checkpoint.")


def param_fingerprint(model: nn.Module, n: int = 8) -> Tensor:
    """A cheap, deterministic snapshot of a FEW MORPH tensors (first `n` by sorted
    name), used to catch a leak into the "frozen" model across the probe run. Stronger
    than trusting `requires_grad` alone: that only guards the ordinary backward path,
    not an in-place op or a shared-storage alias reaching in from outside."""
    items = sorted(model.named_parameters(), key=lambda kv: kv[0])[:n]
    return torch.cat([p.detach().float().flatten()[:256].cpu() for _, p in items])


def assert_disjoint_batches(fit_batches, eval_batches) -> None:
    """Hash every row's raw `input_ids` and confirm no row in `eval_batches` byte-matches
    a row in `fit_batches`. The streaming validation loader should never repeat a row
    within one probe run; this is the cheap, exact check that it didn't."""
    def _hashes(batches):
        out = set()
        for x, _, _ in batches:
            for row in x.cpu():
                out.add(hashlib.sha1(row.numpy().tobytes()).hexdigest())
        return out
    overlap = _hashes(fit_batches) & _hashes(eval_batches)
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} row(s) appear in BOTH fit and eval batches — the "
            f"validation stream must have wrapped, or the two pulls collided. Fit and "
            f"eval batches must be disjoint.")


def decision_band(x: float) -> str:
    if x >= 0.20:
        return "FULL"
    if x <= 0.05:
        return "EMPTY"
    return "INCONCLUSIVE"


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_a2")
    ap.add_argument("--batches", type=int, default=8,
                    help="total eval-stream batches to pull (fit+eval) when "
                         "--fit-batches is not given")
    ap.add_argument("--fit-batches", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--decoder-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2,
                    help="Adam weight decay on the throwaway decoder — a correctness "
                         "guard against memorising fit-set z, not a tuning knob. See "
                         "fit_decoder's docstring.")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    fit_n = a.fit_batches if a.fit_batches is not None else max(1, a.batches - a.eval_batches)
    eval_n = a.eval_batches
    device = torch.device(a.device)

    cfg = build_cfg(a.config, ["training.batch_size=6", "model.use_kernels=false"])
    model, tul_rt = build_model(cfg, device=str(device))
    if tul_rt is None:
        raise RuntimeError(
            f"config {a.config!r} resolves tul.activate_at to 'never' — no TUL "
            f"parameters were built, nothing for this probe to read")
    root = getattr(model, "_orig_mod", model)
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    load_checkpoint(a.ckpt, model, scaler, device)

    model.requires_grad_(False)
    model.eval()
    assert_frozen(model)
    fp_before = param_fingerprint(model)

    loader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="validation", skip_samples=60_000,
        tul=tul_rt.val_data_cfg))

    def pull(n):
        out = []
        for _ in range(n):
            bx, by, bl = next(loader)
            out.append((bx.to(device), by.to(device), bl.to(device)))
        return out

    fit_raw = pull(fit_n)
    eval_raw = pull(eval_n)
    assert_disjoint_batches(fit_raw, eval_raw)
    print(f"  fit: {fit_n} batches, eval: {eval_n} batches (disjoint, verified)\n")

    span_cap = tul_rt.data_cfg.rule.span_cap
    k_model = tul_rt.data_cfg.prefix_k
    vocab_size = int(cfg.model.vocab_size)
    autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
                    else contextlib.nullcontext())

    def extract(batches) -> SlotExamples:
        parts, row_base = [], 0
        for x, y, layout in batches:
            sink: list[Tensor] = []
            with torch.no_grad(), autocast_ctx, capture_prefix_project(root, sink):
                model(x, labels=y, slot_layout=layout)
            values = sink[-1].float()
            S = layout.slot_index.shape[1]
            z = reduce_prefix_values(values, S, k_model)
            parts.append(extract_slot_examples(x, z, layout, span_cap, row_base))
            row_base += x.shape[0]
        return cat_examples(parts)

    fit_ex = extract(fit_raw)
    eval_ex = extract(eval_raw)
    print(f"  usable examples: fit={fit_ex.z.shape[0]} eval={eval_ex.z.shape[0]}  "
          f"(z_dim={fit_ex.z.shape[-1]}, span_cap={span_cap}, prefix_k={k_model})")
    print(f"  excluded slots (last valid per row, or <2 valid): "
          f"fit={fit_ex.n_excluded}/{fit_ex.n_total_valid}  "
          f"eval={eval_ex.n_excluded}/{eval_ex.n_total_valid}\n")

    results = run_four_conditions(
        fit_ex, eval_ex, vocab_size=vocab_size, hidden=a.hidden_dim, steps=a.steps,
        lr=a.lr, batch_size=a.decoder_batch, seed=a.seed, device=device,
        weight_decay=a.weight_decay)

    fp_after = param_fingerprint(model)
    assert torch.equal(fp_before, fp_after), (
        "MORPH parameters changed during the probe — a leak into the checkpoint. "
        "The decoder must never be able to move any MORPH tensor.")

    uf_ip1, n_uf_ip1 = unigram_floor(fit_ex.span_ip1, eval_ex.span_ip1, vocab_size)
    uf_i, n_uf_i = unigram_floor(fit_ex.span_i, eval_ex.span_i, vocab_size)

    print(f"{'condition':<10} {'nats/token':>11} {'eval tok':>10} {'masked':>8} "
          f"{'params':>10} {'train loss':>11}")
    for name in ("PLAN", "SUMMARY", "SHUFFLED", "POSITION"):
        r = results[name]
        print(f"{name:<10} {r['nats_per_token']:>11.4f} {r['n_eval_tokens']:>10} "
              f"{r['n_eval_masked_offsets']:>8} {r['n_decoder_params']:>10} "
              f"{r['final_train_loss']:>11.4f}")
    print(f"{'unigram(i+1)':<10} {uf_ip1:>11.4f} {n_uf_ip1:>10}")
    print(f"{'unigram(i)':<10} {uf_i:>11.4f} {n_uf_i:>10}")

    # ── MEMORIZATION GATE (added 2026-08-28, BEFORE any probe data existed) ──────────
    # `fit_decoder`'s own docstring names the failure this guards: with too little weight
    # decay the decoder MEMORIZES the fit set, its train loss collapses below the marginal
    # entropy, and eval loss on fresh z comes out WORSE than SHUFFLED — which flips the
    # sign of the deciding number. `weight_decay=1e-2` was found on a SYNTHETIC sweep, so
    # nothing yet shows it holds at 49k vocab on real checkpoints.
    #
    # POSITION is the canary and costs nothing extra: it is handed NO z, so it cannot
    # legitimately learn anything z-specific, and every condition shares its decoder size,
    # step budget, LR and seed. Any train/eval gap POSITION shows is memorization capacity
    # that all four conditions had. If the canary sings, REFUSE the readout rather than
    # print a number whose sign cannot be trusted — the `score_arms.py` convention.
    pos = results["POSITION"]
    canary_gap, readable = memorization_gate(pos)
    print(f"\nMEMORIZATION GATE  POSITION eval {pos['nats_per_token']:.4f} - train "
          f"{pos['final_train_loss']:.4f} = {canary_gap:+.4f} nats "
          f"(refuse above {CANARY_MAX:.2f}) -> {'OK' if readable else 'REFUSED'}")
    if not readable:
        print("  POSITION gets NO z, so this gap is pure memorization of the fit targets,")
        print("  and every condition had the same decoder capacity to spend on it. The")
        print("  deciding numbers below are NOT readable. Re-run with a higher")
        print("  --weight-decay or a smaller decoder before quoting either band.")

    # SIGN NOTE (judgment call — see the final report's "judgment calls" section): nats
    # per token is a LOSS, lower is better. `probe-spec.md` writes the deciding
    # quantities as "PLAN - SHUFFLED" and "SUMMARY - PLAN" and requires >=0.20 to read
    # as FULL / "z is a summary" — a POSITIVE number for the informative direction. That
    # only holds if the quantity is actually (worse condition's nats) - (better
    # condition's nats): SHUFFLED nats MINUS PLAN nats (PLAN should be LOWER, i.e.
    # better, when the plan is real), and PLAN nats MINUS SUMMARY nats (SUMMARY should
    # be LOWER when z reads as a summary). Implemented that way below; the printed
    # LABELS keep the spec's own phrasing so the report reads against the spec directly.
    plan_minus_shuffled = results["SHUFFLED"]["nats_per_token"] - results["PLAN"]["nats_per_token"]
    summary_minus_plan = results["PLAN"]["nats_per_token"] - results["SUMMARY"]["nats_per_token"]
    band = decision_band(plan_minus_shuffled)
    # Labels say what is COMPUTED, not what the spec called it. The spec's names
    # ("PLAN - SHUFFLED") have the sign backwards for a LOSS — see the SIGN NOTE above —
    # and a label that disagrees with its own number is how a result gets misread months
    # later. The spec's phrasing is kept in brackets so the bands still line up.
    print(f"\nnats SAVED by the real z   (SHUFFLED - PLAN) = {plan_minus_shuffled:+.4f} "
          f"nats/token  [band: {band}; >=0.20 FULL, <=0.05 EMPTY; "
          f"spec calls this 'PLAN - SHUFFLED']")
    print(f"z is better on its OWN span (PLAN - SUMMARY)  = {summary_minus_plan:+.4f} "
          f"nats/token  [>=0.20 means z reads as a SUMMARY of its own span, not a plan; "
          f"spec calls this 'SUMMARY - PLAN']")
    if not readable:
        print("\n*** BOTH NUMBERS ABOVE ARE REFUSED BY THE MEMORIZATION GATE. ***")

    out = {
        "ckpt": a.ckpt, "config": a.config, "label": a.label,
        "provenance": {
            "fit_batches": fit_n, "eval_batches": eval_n,
            "batch_size": int(cfg.training.batch_size), "seed": a.seed,
            "steps": a.steps, "lr": a.lr, "hidden_dim": a.hidden_dim,
            "decoder_batch": a.decoder_batch, "weight_decay": a.weight_decay,
        },
        "shapes": {
            "z_dim": int(fit_ex.z.shape[-1]), "prefix_k": k_model,
            "span_cap": span_cap, "vocab_size": vocab_size,
            "n_fit_examples": int(fit_ex.z.shape[0]),
            "n_eval_examples": int(eval_ex.z.shape[0]),
            "z_extraction": (
                "TULSlots.prefix_project output [B, S*prefix_k, *mid, C] -> mean over "
                "HC-stream dim (matches _readout's mean reduction) -> concat over "
                "prefix_k -> [B, S, prefix_k*C] one vector per slot"),
        },
        "exclusions": {
            "fit_excluded_slots": fit_ex.n_excluded,
            "fit_total_valid_slots": fit_ex.n_total_valid,
            "eval_excluded_slots": eval_ex.n_excluded,
            "eval_total_valid_slots": eval_ex.n_total_valid,
        },
        "results": results,
        "memorization_gate": {
            "position_eval_nats": pos["nats_per_token"],
            "position_train_loss": pos["final_train_loss"],
            "gap": canary_gap, "max": CANARY_MAX, "readable": readable,
        },
        "unigram_floor": {
            "span_ip1": {"nats_per_token": uf_ip1, "n_tokens": n_uf_ip1},
            "span_i": {"nats_per_token": uf_i, "n_tokens": n_uf_i},
        },
        "deciding_numbers": {
            "plan_minus_shuffled": plan_minus_shuffled, "band": band,
            "summary_minus_plan": summary_minus_plan,
        },
    }
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
