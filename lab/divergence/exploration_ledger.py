"""The exploration ledger: what the fan's SELECTION and SEARCH are worth, in nats, on one
checkpoint, paired token by token against the model's own forward.

Why (Wolfe 2026-09-30, "How can we better prove latent exploration contribution and
effect in these short training runs?"): two 5k runs of one arm differ by about as much as
two arms do, so an arm-vs-arm CE gap cannot price exploration. One checkpoint can: the
same weights and the same rows, one intervention at a time, each per-token CE paired with
the shipped forward's on the SAME tokens. A row bootstrap then reads 0.002 nats.

TERMS.
  shipped   the model's own eval forward (labels present, plan_mode "normal"). Selecting
            arms follow their router; the write-all fans write every cell.
  cell      one of the M fan cells of a slot.
  pick      the cell the coda reads (selecting arms: the router's winner at the exit).
  pass t    loop iteration t = 0 .. T-1 at the arm's deterministic eval depth T.
  delta     reading minus shipped, nats per scored token. POSITIVE = the intervention is
            WORSE than the model, so the component it removes is worth that much.
  quartile  each span's PLAIN model CE (the plain 5k sweep, joined by stream index),
            split into four equal-count bins over spans: Q1 easy .. Q4 hard.

READINGS (only where the arm has the mechanism; "n/a" with the reason otherwise):
  random_exit     the loop as shipped, a uniform random pick (selecting arms).
                  delta > 0: the learned pick beats chance.
  random_search   latent-selected loop: a random winner at EVERY pass and at the exit.
                  delta > 0: the learned search beats a random walk.
  teacher         latent-selected loop: follow the latent teacher at every pass and the
                  exit (`lsel_follow="teacher"`); latent-teacher router: the teacher's
                  exit pick. delta < 0: a better router would buy |delta|.
  cell_i          cell i alone at the exit (every slot forced to i: the val oracle's own
                  "stream i alone" write). single_cell = the mean over i.
  oracle          per span, the cell whose cell_i forward has the lowest summed span CE
                  (the val oracle's per-span minimum, at the token level). Selecting arms:
                  delta <= 0, the headroom of a perfect pick. Write-all fans: the shipped
                  read is ALL cells, so delta > 0 says the full write beats any one cell.
  no_reset        latent-selected loop: no reset to the winner between passes (M
                  independent chains), router pick at the exit. delta > 0: searching
                  from the winner beats picking at the end.
  fixed_lineage   latent-selected loop: follow cell 0 at every pass and read cell 0.
                  delta > 0: selection of any kind is worth that much.
  loser_t{t}      latent-selected loop: at pass t ONLY, follow a random non-winner;
                  router elsewhere. delta > 0 at t: the pick at pass t carries credit.

HOW. Every reading is the LABELLED eval forward (``tul_forward_ablated``; the target
twin computes the teacher), with the coda state captured at ``_tul_group_losses`` and
turned into logits exactly as the label-free path does (``embed.attend`` with the slot id
masked). No ``morph/model`` code changes: each intervention is an instance-level patch
installed by :func:`ledger_patch` and removed in a ``finally``:
  * the latent-selected loop's per-pass pick: ``tul_fan_lsel_router.select`` (called once
    per pass by ``_lsel_pass``; at eval the loop follows its winner on every slot);
  * the reset: ``morph.model.transformer.reset_to_winner`` -> identity;
  * a router's exit pick: ``_tul_fan_route``'s returned winner;
  * a write-all fan's write: the ``cells=`` argument of ``tul.prefix_project``;
  * the eval-only oracle (K extra coda passes): a no-op except on the shipped forward,
    where its ``fan_oracle_ce`` is kept as a cross-check of :func:`oracle_tokens`.

SELF-CHECKS (in the JSON): the captured CE vs the forward's own ``ce_tokens`` per batch,
and the labelled capture vs the label-free forward's logits on the first batch.

Usage (GPU):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/exploration_ledger.py \\
      --ckpt lsel_joint=tul_slot_spandec_strict_fan4_all_fp01_lsel_joint=.../step_5000.pt \\
      --plain_tokens .../sweep_plain-panel-norm-match_5000.plain-panel-norm-match.tokens.npz \\
      --rows 480 --batch 3 --out .../ledger_lsel_joint.json
Note: .agents/notes/proposed/testing/2026-09-30-exploration-ledger.md
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _build import ROOT, build_cfg, parse_ckpt_spec  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from _stats import paired_bootstrap_ci  # noqa: E402

N_BOOT = 2000
_MISSING = object()
ANCHORS = {
    "shipped": "the model's own eval forward (the reference every delta is taken against)",
    "random_exit": "0 = the learned pick is no better than a random cell; > 0 = pick value",
    "random_search": "0 = the learned search is no better than a random walk; > 0 = search "
                     "value",
    "teacher": "< 0 = following the latent teacher beats the router (router error); "
               "0 = the router matches the teacher",
    "single_cell": "write-all fans: > 0 = reading all cells beats one (width value); "
                   "selecting arms: > 0 = the pick beats an arbitrary fixed cell",
    "oracle": "selecting arms (the shipped read IS one cell): <= 0, |delta| = headroom of "
              "a perfect per-span pick. Write-all fans (the shipped read is all cells): "
              "> 0 = reading every cell beats the best single cell per span",
    "no_reset": "> 0 = resetting to the winner each pass beats M independent chains "
                "picked at the exit; 0 = the per-pass search adds nothing",
    "fixed_lineage": "> 0 = selection of any kind is worth that much; 0 = selection is "
                     "worthless",
    "loser": "> 0 at pass t = the pick at pass t carries credit to the final CE; 0 = that "
             "pass's pick does not matter",
}


# ── arm discovery ────────────────────────────────────────────────────────────────────


def arm_kind(model) -> str:
    """``"lsel"`` (the latent-selected loop), ``"reader"`` / ``"latent"`` (the two
    routers), or ``"write_all"`` (the ungraded, coda-graded and factor fans). Anything
    else is refused: the ledger prices a fan's cells and has nothing to read otherwise."""
    if getattr(model, "_code_enum_k", 0):
        raise SystemExit("exploration ledger: code rollouts (tul.code_enum_k > 1) are not "
                         "supported; the four-rollout ensemble has no per-cell pick.")
    if getattr(model, "_lsel_mode", "off") != "off":
        # `tul.fan_lsel_read: all`: the loop selects, the coda reads every candidate
        return "lsel_all" if getattr(model, "_lsel_read_all", False) else "lsel"
    if getattr(model, "tul_fan_router", None) is not None:
        return str(model.cfg.tul.fan_route)
    fan = getattr(model, "tul_fan", None)
    if fan is not None and fan.mode == "all":
        return "write_all"
    raise SystemExit("exploration ledger: not a write-all fan arm (tul.fan_k >= 2, "
                     "fan_mix 'all')")


def fan_m(model) -> int:
    return int(model.cfg.tul.fan_k)


# ── the interventions ────────────────────────────────────────────────────────────────


def _rand_cells(shape, m: int, gen: torch.Generator, device) -> torch.Tensor:
    return torch.randint(0, m, tuple(shape), generator=gen).to(device)


def _rand_losers(winner: torch.Tensor, m: int, gen: torch.Generator) -> torch.Tensor:
    """A uniform random cell OTHER than ``winner``, per slot."""
    r = torch.randint(0, m - 1, tuple(winner.shape), generator=gen).to(winner.device)
    return (winner + 1 + r) % m


@contextlib.contextmanager
def ledger_patch(model, *, select_fn=None, exit_fn=None, write_cell: int | None = None,
                 no_reset: bool = False, keep_oracle: bool = False, counter: list | None = None):
    """Install one intervention on ``model`` for the duration of the block.

    ``select_fn(t, winner) -> winner``: the latent-selected loop's pick at pass ``t``
    (``None`` keeps the router's). ``exit_fn(route, tgt) -> winner``: a router arm's exit
    pick. ``write_cell``: a write-all fan writes cell ``i`` alone (the others blank, the
    val oracle's "stream i alone"). ``no_reset``: the latent-selected loop's reset is the
    identity. ``keep_oracle``: leave the eval-only val oracle running (default: no-op).
    ``counter``: a list that receives one entry per pass the selection ran (the pass
    count of each forward)."""
    import morph.model.transformer as T

    installed: list[tuple[object, str, object]] = []
    old_reset = T.reset_to_winner

    def _install(obj, name: str, fn) -> None:
        # remember an instance-level value already there (a test spy), restore it after
        installed.append((obj, name, obj.__dict__.get(name, _MISSING)))
        obj.__dict__[name] = fn
    try:
        if not keep_oracle:
            _install(model, "_tul_fan_oracle", lambda *a, **k: None)
        router = getattr(model, "tul_fan_lsel_router", None)
        if router is not None and (select_fn is not None or counter is not None):
            orig_select = router.select
            calls = [0]

            def _select(scores):
                winner, p = orig_select(scores)
                t = calls[0]
                calls[0] += 1
                if counter is not None:
                    counter.append(t)
                if select_fn is not None:
                    winner = select_fn(t, winner)
                return winner, p
            _install(router, "select", _select)
        elif select_fn is not None:
            raise ValueError("select_fn needs a latent-selected loop model")
        if exit_fn is not None:
            if getattr(model, "tul_fan_router", None) is None:
                raise ValueError("exit_fn needs a router arm (tul.fan_route)")
            orig_route = model._tul_fan_route

            def _route(cells, xn, layout, tgt, stats):
                route = dict(orig_route(cells, xn, layout, tgt, stats))
                route["winner"] = exit_fn(route, tgt)
                return route
            _install(model, "_tul_fan_route", _route)
        if write_cell is not None:
            i = int(write_cell)
            orig_pp = model.tul.prefix_project

            def _pp(h_slots, layout, l_total, cells=None):
                if cells is None:
                    raise RuntimeError("write_cell: the write carried no cells (not a "
                                       "write-all fan forward)")
                blank = torch.zeros_like(cells)
                blank[:, :, i] = cells[:, :, i]
                return orig_pp(h_slots, layout, l_total, cells=blank)
            _install(model.tul, "prefix_project", _pp)
        if no_reset:
            if getattr(model, "_lsel_mode", "off") == "off":
                raise ValueError("no_reset needs a latent-selected loop model")
            T.reset_to_winner = lambda h, winner, reset, m: h
        yield
    finally:
        T.reset_to_winner = old_reset
        for obj, name, prior in reversed(installed):
            if prior is _MISSING:
                obj.__dict__.pop(name, None)
            else:
                obj.__dict__[name] = prior


@torch.no_grad()
def ce_map(model, inp, labels, layout, device, lsel_follow: str | None = None
           ) -> tuple[torch.Tensor, dict]:
    """Per-position CE ``[B, L]`` (cpu fp32) of the LABELLED eval forward, plus its scalar
    outputs. The coda state reaching ``_tul_group_losses`` is turned into logits the way
    the label-free path does (``embed.attend``, slot id masked), under the same autocast.
    Exactly one capture per forward or this raises."""
    held: list[torch.Tensor] = []
    orig = model._tul_group_losses
    slot_id = int(model.cfg.tul.slot_id)

    def _cap(x, labels_, layout_, want_groups=True):
        held.append(model.embed.attend(x).index_fill(
            -1, torch.tensor([slot_id], device=x.device), float("-inf")))
        return orig(x, labels_, layout_, want_groups=want_groups)
    prior = model.__dict__.get("_tul_group_losses", _MISSING)
    model.__dict__["_tul_group_losses"] = _cap
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            kw = {} if lsel_follow is None else {"lsel_follow": lsel_follow}
            out = model.tul_forward_ablated(inp.to(device), labels.to(device), layout,
                                            plan_mode="normal", **kw)
    finally:
        if prior is _MISSING:
            model.__dict__.pop("_tul_group_losses", None)
        else:
            model.__dict__["_tul_group_losses"] = prior
    if len(held) != 1:
        raise RuntimeError(f"ce_map: {len(held)} coda captures in one forward (expected 1)")
    logits = held[0].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    scal = {k: float(v) for k, v in out.items()
            if torch.is_tensor(v) and v.dim() == 0 and (k == "ce_tokens" or k.startswith("fan_"))}
    return ce.cpu(), scal


# ── the reading plan ─────────────────────────────────────────────────────────────────


def _gen(name: str, batch_idx: int, seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed((seed * 1_000_003 + batch_idx * 7919 + zlib.crc32(name.encode())) % (2**63))
    return g


def reading_plan(kind: str, m: int, passes: int) -> list[tuple[str, dict]]:
    """``[(name, spec)]`` where ``spec`` holds a builder ``make(gen) -> patch kwargs`` and
    optional ``follow``. ``passes`` = T, the latent-selected loop's eval pass count."""
    plan: list[tuple[str, dict]] = [("shipped", {"make": lambda g: {}, "keep_oracle": True})]
    last = passes - 1
    if kind in ("lsel", "lsel_all"):
        if kind == "lsel":
            plan.append(("random_exit", {"make": lambda g: {"select_fn": (
                lambda t, w: _rand_cells(w.shape, m, g, w.device) if t == last else w)}}))
        plan.append(("random_search", {"make": lambda g: {"select_fn": (
            lambda t, w: _rand_cells(w.shape, m, g, w.device))}}))
        plan.append(("teacher", {"make": lambda g: {}, "follow": "teacher"}))
        for i in range(m):
            if kind == "lsel":
                plan.append((f"cell_{i}", {"make": (lambda i: lambda g: {"select_fn": (
                    lambda t, w: torch.full_like(w, i) if t == last else w)})(i)}))
            else:   # the coda reads every candidate: cell i alone is the write-all blank
                plan.append((f"cell_{i}", {"make": (lambda i: lambda g: {"write_cell": i})(i)}))
        plan.append(("no_reset", {"make": lambda g: {"no_reset": True}}))
        plan.append(("fixed_lineage", {"make": lambda g: {"select_fn": (
            lambda t, w: torch.zeros_like(w))}}))
        for tt in range(passes - 1):
            plan.append((f"loser_t{tt}", {"make": (lambda tt: lambda g: {"select_fn": (
                lambda t, w: _rand_losers(w, m, g) if t == tt else w)})(tt)}))
    elif kind in ("reader", "latent"):
        plan.append(("random_exit", {"make": lambda g: {"exit_fn": (
            lambda r, tgt: _rand_cells(r["winner"].shape, m, g, r["winner"].device))}}))
        if kind == "latent":
            def _teach(r, tgt):
                if r.get("teacher") is None or tgt is None:
                    raise RuntimeError("teacher reading: the route carries no teacher "
                                       "(no labels / no target twin)")
                return torch.where(tgt["ok"], r["teacher"], r["winner"])
            plan.append(("teacher", {"make": lambda g: {"exit_fn": _teach}}))
        for i in range(m):
            plan.append((f"cell_{i}", {"make": (lambda i: lambda g: {"exit_fn": (
                lambda r, tgt: torch.full_like(r["winner"], i))})(i)}))
    elif kind == "write_all":
        for i in range(m):
            plan.append((f"cell_{i}", {"make": (lambda i: lambda g: {"write_cell": i})(i)}))
    else:
        raise ValueError(kind)
    return plan


def not_applicable(kind: str) -> dict[str, str]:
    why_sel = "the arm has no pick: the coda reads every cell"
    why_loop = "only the latent-selected loop selects inside the loop"
    na: dict[str, str] = {}
    if kind == "write_all":
        na |= {"random_exit": why_sel, "teacher": why_sel}
    if kind == "lsel_all":
        na["random_exit"] = "the coda reads every final candidate: there is no exit pick"
    if kind == "reader":
        na["teacher"] = "the reader-trained router has no latent teacher"
    if kind not in ("lsel", "lsel_all"):
        na |= {k: why_loop for k in ("random_search", "no_reset", "fixed_lineage", "loser")}
    return na


# ── statistics ───────────────────────────────────────────────────────────────────────


def oracle_tokens(cell_ce: np.ndarray, span_key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``cell_ce`` ``[M, N]`` per-token CE of each cell_i forward, ``span_key`` ``[N]`` the
    token's span id. Per span, the cell with the lowest SUMMED span CE; returns that
    cell's per-token CE ``[N]`` and the per-token chosen cell ``[N]``."""
    m = cell_ce.shape[0]
    _, inv = np.unique(span_key, return_inverse=True)
    n_span = int(inv.max()) + 1 if inv.size else 0
    sums = np.stack([np.bincount(inv, weights=cell_ce[i], minlength=n_span) for i in range(m)])
    best = sums.argmin(axis=0)                     # [n_span]; ties -> the lowest index
    pick = best[inv]
    return cell_ce[pick, np.arange(cell_ce.shape[1])], pick


def span_quartiles(plain_tok: np.ndarray, span_key: np.ndarray) -> np.ndarray:
    """Per-token quartile id 0..3 of its span's mean PLAIN CE (equal-count bins over the
    spans with at least one plain-matched token); -1 where the span has none."""
    ok = np.isfinite(plain_tok)
    _, inv = np.unique(span_key, return_inverse=True)
    n_span = int(inv.max()) + 1 if inv.size else 0
    s = np.bincount(inv, weights=np.where(ok, plain_tok, 0.0), minlength=n_span)
    c = np.bincount(inv, weights=ok.astype(np.float64), minlength=n_span)
    has = c > 0
    mean = np.full(n_span, np.nan)
    mean[has] = s[has] / c[has]
    q = np.full(n_span, -1, dtype=np.int64)
    if has.any():
        edges = np.quantile(mean[has], [0.25, 0.5, 0.75])
        q[has] = np.searchsorted(edges, mean[has], side="right")
    return q[inv]


def delta_ci(reading: np.ndarray, shipped: np.ndarray, row: np.ndarray, n_rows: int,
             mask: np.ndarray | None = None, seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """Row-bootstrap CI of mean(reading - shipped) over the (masked) tokens."""
    w = np.ones_like(reading, dtype=np.float64) if mask is None else mask.astype(np.float64)
    sa = np.bincount(row, weights=reading * w, minlength=n_rows)
    sb = np.bincount(row, weights=shipped * w, minlength=n_rows)
    cn = np.bincount(row, weights=w, minlength=n_rows)
    keep = cn > 0
    if not keep.any():
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_units": 0, "n_tokens": 0}
    r = paired_bootstrap_ci(sa[keep], sb[keep], cn[keep], n_boot=n_boot, seed=seed)
    r["n_tokens"] = int(cn.sum())
    return r


# ── one arm ──────────────────────────────────────────────────────────────────────────


def ledger_arm(model, batches, device: str, plain: tuple[np.ndarray, np.ndarray] | None,
               seed: int = 0, n_boot: int = N_BOOT, tol: float = 1e-2,
               log=print) -> tuple[dict, dict[str, np.ndarray]]:
    """Run every reading of one arm on ``batches`` ``[(inp, labels, layout, idx)]``.
    Returns ``(json_block, token_arrays)``."""
    kind = arm_kind(model)
    m = fan_m(model)
    # token bookkeeping, once
    tok_row, tok_key, tok_idx, tok_masks = [], [], [], []
    r0 = 0
    for inp, labels, layout, idx in batches:
        S1 = int(layout.slot_valid.shape[1]) + 1
        tokpos = (~layout.slot_mask.cpu()) & (labels >= 0)
        rows = torch.arange(inp.shape[0]).view(-1, 1).expand_as(tokpos) + r0
        bag = layout.bag_id.cpu().clamp_min(0)
        tok_row.append(rows[tokpos].numpy())
        tok_key.append((rows * S1 + bag)[tokpos].numpy())
        tok_idx.append(idx[tokpos].numpy())
        tok_masks.append(tokpos)
        r0 += inp.shape[0]
    n_rows = r0
    row = np.concatenate(tok_row).astype(np.int64)
    key = np.concatenate(tok_key).astype(np.int64)
    sidx = np.concatenate(tok_idx).astype(np.int64)

    # the pass count of the latent-selected loop's eval forward
    passes = 0
    if kind in ("lsel", "lsel_all"):
        cnt: list = []
        inp, labels, layout, _ = batches[0]
        with ledger_patch(model, counter=cnt):
            ce_map(model, inp, labels, layout.to(device), device)
        passes = len(cnt)
        if passes < 1:
            raise RuntimeError("latent-selected loop: the selection never ran")
    plan = reading_plan(kind, m, passes)
    log(f"  kind={kind} M={m} passes={passes} readings={[n for n, _ in plan]} "
        f"rows={n_rows} tokens={row.size}")

    toks: dict[str, np.ndarray] = {}
    checks: dict = {"ce_tokens_max_abs_dev": 0.0}
    shipped_scal: dict = {}
    t0 = time.time()
    for name, spec in plan:
        ces = []
        for bi, ((inp, labels, layout, _), tokpos) in enumerate(zip(batches, tok_masks)):
            g = _gen(name, bi, seed)
            kw = spec["make"](g)
            cnt = [] if kind in ("lsel", "lsel_all") else None
            with ledger_patch(model, keep_oracle=spec.get("keep_oracle", False),
                              counter=cnt, **kw):
                ce, scal = ce_map(model, inp, labels, layout.to(device), device,
                                  lsel_follow=spec.get("follow"))
            if cnt is not None and len(cnt) != passes:
                raise RuntimeError(f"{name} batch {bi}: {len(cnt)} passes, expected {passes}")
            ces.append(ce[tokpos].numpy().astype(np.float64))
            if "ce_tokens" in scal:
                dev = abs(float(ce[tokpos].mean()) - scal["ce_tokens"])
                checks["ce_tokens_max_abs_dev"] = max(checks["ce_tokens_max_abs_dev"], dev)
                if dev > tol:
                    raise RuntimeError(f"{name} batch {bi}: captured CE {float(ce[tokpos].mean()):.5f}"
                                       f" vs the forward's ce_tokens {scal['ce_tokens']:.5f}")
            if name == "shipped":
                for k, v in scal.items():
                    shipped_scal.setdefault(k, []).append(v)
        toks[name] = np.concatenate(ces)
        log(f"  {name:14s} ce={toks[name].mean():.4f}  ({time.time() - t0:.0f}s)")

    # the label-free forward reproduces the labelled capture (first batch)
    inp, labels, layout, _ = batches[0]
    with ledger_patch(model):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=device == "cuda"):
            lf = model.tul_forward_ablated(inp.to(device), None, layout.to(device),
                                           plan_mode="normal")["logits"].float()
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    ce_lf = F.cross_entropy(lf.reshape(-1, lf.shape[-1]), lab.reshape(-1),
                            reduction="none").reshape(lab.shape).cpu()
    n0 = int(tok_masks[0].sum())
    checks["label_free_max_abs_dev_batch0"] = float(
        np.abs(ce_lf[tok_masks[0]].double().numpy() - toks["shipped"][:n0]).max())

    # derived readings
    cells = np.stack([toks[f"cell_{i}"] for i in range(m)])
    toks["single_cell"] = cells.mean(axis=0)
    toks["oracle"], pick = oracle_tokens(cells, key)

    # plain join -> quartiles
    quart = None
    plain_info: dict = {"used": False}
    if plain is not None:
        # Shift 0 BY CONSTRUCTION: `pack_rows` gives both cuts the stream index of the INPUT
        # token at a scored position (label = stream[idx + 1]), the plain sweep's npz
        # included. The correlation by shift is kept as a diagnostic only: on an untrained
        # checkpoint (CE ~10) it is too weak to decide anything (0.31 at step 45).
        pidx, pce = plain
        corr = {}
        for sh in (-1, 0, 1):
            _, a_, b_ = np.intersect1d(sidx, pidx + sh, return_indices=True)
            corr[sh] = (float(np.corrcoef(toks["shipped"][a_], pce[b_])[0, 1])
                        if a_.size > 1000 else float("nan"))
        _, a, b = np.intersect1d(sidx, pidx, return_indices=True)
        ptok = np.full(sidx.shape, np.nan)
        ptok[a] = pce[b]
        quart = span_quartiles(ptok, key)
        plain_info = {"used": True, "shift": 0, "corr_by_shift": corr,
                      "shift0_is_best_corr": bool(corr[0] >= max(corr[-1], corr[1])),
                      "matched_tokens": int(a.size),
                      "plain_mean_on_matched": float(pce[b].mean()),
                      "shipped_minus_plain_on_matched": float(
                          (toks["shipped"][a] - pce[b]).mean())}

    shipped = toks["shipped"]
    readings: dict = {}
    for name in toks:
        if name == "shipped":
            continue
        anchor = ANCHORS.get("loser" if name.startswith("loser_t") else
                             ("single_cell" if name.startswith("cell_") else name), "")
        r = {"ce": float(toks[name].mean()), "anchor": anchor,
             "delta": delta_ci(toks[name], shipped, row, n_rows, seed=seed, n_boot=n_boot)}
        if quart is not None:
            r["by_plain_quartile"] = {
                f"Q{q + 1}": delta_ci(toks[name], shipped, row, n_rows, mask=(quart == q),
                                      seed=seed, n_boot=n_boot) for q in range(4)}
        readings[name] = r
    oracle_model = shipped_scal.get("fan_oracle_ce")
    checks["oracle_tokens_mean"] = float(toks["oracle"].mean())
    if oracle_model:
        checks["fan_oracle_ce_model_mean"] = float(np.mean(oracle_model))
        checks["fan_oracle_ce_note"] = ("the model's oracle scores bags >= 1 only, with no "
                                        "slot-id mask; the ledger's oracle scores every "
                                        "token with the mask, so a small gap is expected")
    block = {
        "kind": kind, "M": m, "passes": passes, "rows": n_rows, "n_tokens": int(row.size),
        "shipped_ce": float(shipped.mean()), "readings": readings,
        "not_applicable": not_applicable(kind), "anchors": ANCHORS,
        "oracle_pick_share": [float((pick == i).mean()) for i in range(m)],
        "plain": plain_info, "self_checks": checks,
        "shipped_fan_stats": {k: float(np.mean(v)) for k, v in shipped_scal.items()},
        "wall_s": round(time.time() - t0, 1),
    }
    toks_out = {"tok_index": sidx, "row": row, "span_key": key, **toks}
    if quart is not None:
        toks_out["plain_quartile"] = quart
    return block, toks_out


def _fmt(d: dict) -> str:
    return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH[=OVR]")
    ap.add_argument("--plain_tokens", default=None,
                    help="the plain model's core_depth_sweep tokens npz (ce_6 is read)")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    plain = None
    if a.plain_tokens:
        z = np.load(a.plain_tokens)
        plain = (z["tok_index"].astype(np.int64), z["ce_6"].astype(np.float64))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res: dict = {}
    for spec in a.ckpt:
        label, config, path, ovr = parse_ckpt_spec(spec)
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        rt = build_tul_runtime(cfg)
        if rt is None:
            raise SystemExit(f"{label}: not a TUL arm")
        model, step = load_ckpt(cfg, path, a.device, rt.model_cfg)
        model.eval()
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        row_tokens = rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
        stream = stream_from_loader(loader, a.rows * row_tokens)
        batches = pack_rows(stream, rt, cfg, a.batch, False)[:-(-a.rows // a.batch)]
        print(f"[{label}] {path} step {step}", flush=True)
        block, toks = ledger_arm(model, batches, a.device, plain, seed=a.seed,
                                 n_boot=a.n_boot, tol=a.tol,
                                 log=lambda s: print(s, flush=True))
        block |= {"config": config, "path": path, "step": step, "overrides": ovr}
        npz = a.out.rsplit(".", 1)[0] + f".{label}.tokens.npz"
        np.savez_compressed(npz, **{k: (v.astype(np.float32) if v.dtype == np.float64 else v)
                                    for k, v in toks.items()})
        block["tokens_npz"] = npz
        res[label] = block
        print(f"[{label}] shipped ce {block['shipped_ce']:.4f}  (delta = reading - shipped)")
        for name, r in block["readings"].items():
            q = ""
            if "by_plain_quartile" in r:
                q = "  Q1..Q4 " + " ".join(f"{v['point']:+.4f}"
                                           for v in r["by_plain_quartile"].values())
            print(f"    {name:14s} {_fmt(r['delta'])}{q}")
        print(f"    self-checks {block['self_checks']}")
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
