"""RANK ANATOMY of the TUL slot state — shared offset, or a true collapse?

The strict ruler `slot-spandec-strict` logs `val/slot_eff_rank` 5.76 and
`val/slot_pairwise_cos` 0.71 over the 64 slot states of a row in 1024 dimensions. Both
numbers are read off ONE stage (the loop's exit, through `_readout`) and neither says
WHICH of two very different geometries produced it:

  SHARED OFFSET   one common direction carries almost all of the energy and the states
                  differ underneath it at high rank. `val/slot_eff_rank` is already the
                  CENTERED participation ratio (`morph/model/fm_planner.py::effective_rank`
                  subtracts the mean of the pooled set), so a shared offset does NOT
                  explain 5.76 on its own — but the PER-ROW mean is not what the pooled
                  statistic removes, and a per-row offset is exactly what a reader that
                  sees one row at a time would have to subtract.
  TRUE COLLAPSE   the residual is low rank too: after the row mean comes out there are
                  still only a handful of directions, and the register (M cells per slot)
                  or any other widening has nothing to widen.

This instrument reports BOTH, per row and pooled, at every stage the state passes
through, so the reader can see WHERE the rank is lost: the seed, the prelude, the loop's
passes, or the prefix write the coda reads.

STAGES (per row, over its valid slots):

  s0_seed   the slot's input value, `TULSlots.slot_input(embed(ids), layout,
            add_e_slot=True)` at the slot positions — captured by spying on the model's
            own call inside `_tul_front`, so it is the seed the prelude actually got.
            PRE-prelude, so it lives in embedding space, not in the carrier.
  s1_entry  the loop entry `core_init(e)` — captured with a forward hook on
            `model.core_init`, the state `_tul_core` starts iterating from.
  pass1..T  the state after each pass. `db_traj` is NOT available here: `_tul_core`
            builds it only for a training-only knob (`oracle_z`, `spandec_per_pass`,
            `db_loop`, the staged/every-pass MUX, the critic) or for
            `prefix_source in ("trajectory", "entry_exit")` — an eval forward on the
            strict family keeps it None. So each pass is read as the EXIT OF A FORCED
            DEPTH (`_tul_core(slot_depths=d)`, `_slot_depth_override`). The loop's
            update is a deterministic function of the previous state at eval, so the
            exit at forced depth d IS the trajectory's entry d; the instrument checks
            that identity at d == T against the unforced run.
  sX_exit   the exit at the checkpoint's OWN eval depth (no forced table). Asserted
            bit-equal to passT when T is that depth.
  sW_write  the OUTPUT of `TULSlots.prefix_project(h_exit, layout, l_total)` — the
            `prefix_k` cell values written into the row, i.e. what the coda reads at
            those positions. `prefix_k` cells per slot, so this stage has K times as
            many vectors per row as the others.

TWO VIEWS of every stage:

  raw       the Hyper-Connection stream MEAN, `[.., C]`, no normalisation. This is the
            state's own geometry, norms included.
  readout   `final_norm(lm_mixer(raw))` — identical to `MORPHTransformer._readout` on a
            carrier stage (asserted, bit-exact), which is the readout `val/slot_eff_rank`
            is computed on and the one the MUX head and the span decoder read. Every
            vector comes out of an RMSNorm, so norms are ~constant by construction and
            the readout view is a pure DIRECTION geometry.
            s0_seed is pre-prelude and is not a carrier state; its readout view applies
            the same map for chain continuity and is NOT the coda's space. Say so when
            quoting it.

PER-ROW NUMBERS (median and IQR over rows), and the same pooled over every valid slot of
every row:

  eff_rank_raw        participation ratio (Σλ)²/Σλ² of the UNCENTERED second moment
  eff_rank_centered   the same after subtracting the row's (or the pool's) mean — the
                      definition `val/slot_eff_rank` uses
  eff_rank_unit       the same on the per-slot unit-normalised states (cosine geometry)
  pc1_frac_raw/_cent  the top component's share of the spectrum
  cos_raw/_centered   mean cosine between distinct slot pairs, before and after the mean
  norm_mean           mean ‖s‖
  row_mean_norm       ‖mean(s)‖, and its ratio to norm_mean — the shared-offset size

  resid_prev          ‖s − P s‖/‖s‖ where P projects on the PREVIOUS stage's top-8 right
                      singular vectors (uncentered, per row). Where new directions enter.

REPRODUCTION CHECK. Every batch is also scored by the model's OWN
`tul_slot_state_probe`, the exact code the trainer logs `val/slot_eff_rank` from. The
instrument's sX_exit readout pooled-per-batch number must land on it. If it does not,
the instrument is reading a different stage or a different readout and NOTHING else in
its output should be believed.

WHAT IT CANNOT SAY. Eval-only, on a trained checkpoint: a stage that reads low rank says
this model's states sit there, not that a model trained differently would. The rows come
from the probe loader (`skip_samples=0`), not the trainer's held-out shard
(`skip_samples=50_000`), and the batch is the probe's, so the reproduced number is close
to but not identical with the run's logged one.

Usage:
  python lab/divergence/slot_rank_anatomy.py \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
      --rows 480 --batch 3 --device cuda --out results/strict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")

TOP_K_SPAN = 8          # the subspace `resid_prev` projects onto


# ── the statistics ───────────────────────────────────────────────────────────

def spectrum(m: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of ``mᵀm`` (the uncentered second moment), descending, float64.

    Routed through whichever side is smaller: ``svdvals`` for the per-row case (64 slots
    in 1024 dims) and the Gram matrix for the pooled case (30k slots). Both give the same
    non-zero spectrum, and the participation ratio is scale-free, so the missing 1/n is
    irrelevant and deliberately not applied.
    """
    m = m.double()
    if m.shape[0] <= m.shape[1]:
        return torch.linalg.svdvals(m).pow(2)
    g = m.T @ m
    return torch.linalg.eigvalsh(g).clamp_min(0.0)


def participation_ratio(lam: torch.Tensor) -> float:
    """``(Σλ)² / Σλ²`` — the definition `morph/model/fm_planner.py::effective_rank` uses.

    ``d`` for an isotropic cloud, 1 for a rank-1 one, 0 for a set with no spread at all.
    """
    s1 = float(lam.sum())
    s2 = float((lam * lam).sum())
    if s2 <= 0.0:
        return 0.0
    return s1 * s1 / s2


def mean_pairwise_cos(m: torch.Tensor) -> float:
    """Mean cosine between DISTINCT rows, exactly `fm_planner.mean_pairwise_cos`.

    Computed from the sum of the unit vectors — ``(‖Σu‖² − n) / (n(n−1))`` — so the pooled
    case never builds an n×n Gram matrix.
    """
    n = m.shape[0]
    if n < 2:
        return 0.0
    u = m.double()
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    s = u.sum(0)
    return float((float(s.dot(s)) - n) / (n * (n - 1)))


def rank_stats(m: torch.Tensor) -> dict:
    """Every per-set number this instrument reports, for one set of states ``[n, d]``."""
    m = m.double()
    n, d = int(m.shape[0]), int(m.shape[1])
    if n < 2:
        return {"n": n, "d": d}
    lam = spectrum(m)
    mu = m.mean(0)
    mc = m - mu
    lam_c = spectrum(mc)
    norms = m.norm(dim=1)
    unit = m / norms.clamp_min(1e-12).unsqueeze(1)
    s_raw, s_cen = float(lam.sum()), float(lam_c.sum())
    return {
        "n": n, "d": d,
        "eff_rank_raw": participation_ratio(lam),
        "eff_rank_centered": participation_ratio(lam_c),
        "eff_rank_unit": participation_ratio(spectrum(unit)),
        "pc1_frac_raw": float(lam.max()) / s_raw if s_raw > 0 else 0.0,
        "pc1_frac_centered": float(lam_c.max()) / s_cen if s_cen > 0 else 0.0,
        "cos_raw": mean_pairwise_cos(m),
        "cos_centered": mean_pairwise_cos(mc),
        "norm_mean": float(norms.mean()),
        "row_mean_norm": float(mu.norm()),
        "row_mean_norm_ratio": float(mu.norm()) / max(float(norms.mean()), 1e-12),
    }


def residual_fraction(prev: torch.Tensor, cur: torch.Tensor, k: int = TOP_K_SPAN) -> float:
    """Mean ``‖s − P s‖ / ‖s‖`` over the rows of ``cur``, ``P`` = top-``k`` span of ``prev``.

    The span is UNCENTERED: the question is "how much of this stage's state is a
    direction the previous stage already had", and a shared offset is one of those
    directions. 0 = the stage added nothing outside the old subspace, 1 = it is entirely
    new.
    """
    prev, cur = prev.double(), cur.double()
    kk = min(int(k), int(min(prev.shape)))
    v = torch.linalg.svd(prev, full_matrices=False).Vh[:kk]          # [k, d]
    proj = (cur @ v.T) @ v
    return float(((cur - proj).norm(dim=1) / cur.norm(dim=1).clamp_min(1e-12)).mean())


def quantiles(vals: list[float]) -> dict:
    """Median and IQR of a per-row series."""
    if not vals:
        return {}
    t = torch.tensor(vals, dtype=torch.float64)
    return {"median": float(t.median()), "q25": float(t.quantile(0.25)),
            "q75": float(t.quantile(0.75)), "mean": float(t.mean()), "n_rows": len(vals)}


# ── the capture ──────────────────────────────────────────────────────────────

def readout_view(model, raw: torch.Tensor) -> torch.Tensor:
    """``final_norm(lm_mixer(raw))`` — the tail of :meth:`MORPHTransformer._readout`.

    Taken on the ALREADY stream-averaged state so the same expression serves the carrier
    stages and the pre-prelude seed, which has no stream axis. Asserted bit-equal to
    ``model._readout`` on a carrier stage in :func:`capture_batch`.
    """
    return model.final_norm(model.lm_mixer(raw))


def views(model, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(raw, readout)`` for a stage tensor ``[B, n_vec, streams, C]`` or ``[B, n_vec, C]``."""
    raw = t.mean(dim=2) if t.dim() == 4 else t
    return raw, readout_view(model, raw)


@torch.no_grad()
def capture_batch(model, inp, layout, depth: int, check_exit: bool = True
                  ) -> tuple[dict, dict]:
    """``({stage: {"raw": [B,n,C], "readout": [B,n,C]}}, {stage: valid [B,n]})`` for one batch.

    Everything runs inside ONE autocast region, at the model's own relation
    (`_tul_tg_kwargs`) — a bare `_tul_front` scores a strict arm from an unrestricted
    prelude and every state downstream of it is off-distribution (2026-09-13).
    """
    from morph.model.tul import gather_valid

    fkw, freset, _ckw, _creset = model._tul_tg_kwargs(layout)
    stages: dict[str, torch.Tensor] = {}

    # ── s0: the slot seed, spied off the model's own `slot_input` call ──────────
    seed: list[torch.Tensor] = []
    real_slot_input = model.tul.slot_input

    def spy(signal, lay, add_e_slot):
        out = real_slot_input(signal, lay, add_e_slot)
        if add_e_slot:
            seed.append(out)
        return out

    model.tul.slot_input = spy
    try:
        x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw,
                                         ret_reset_mask=freset)
    finally:
        model.tul.slot_input = real_slot_input
    if len(seed) != 1:
        raise RuntimeError(f"_tul_front made {len(seed)} add_e_slot=True slot_input calls, "
                           "expected exactly 1 — the seed capture is ambiguous")
    stages["s0_seed"] = gather_valid(seed[0], layout.slot_index, layout.slot_valid)

    # ── s1: the loop entry, hooked on the model's own `core_init` ───────────────
    entry: list[torch.Tensor] = []
    handle = model.core_init.register_forward_hook(lambda _m, _i, o: entry.append(o))
    try:
        table = torch.full(layout.slot_index.shape, 1, dtype=torch.long, device=inp.device)
        out = model._tul_core(x, x0, bigram, layout, input_ids=inp, slot_depths=table)
    finally:
        handle.remove()
    if len(entry) != 1:
        raise RuntimeError(f"_tul_core called core_init {len(entry)} times, expected 1")
    stages["s1_entry"] = entry[0]
    stages["pass1"] = out[1]

    # ── pass 2..T: the exit at each forced depth ────────────────────────────────
    for d in range(2, depth + 1):
        table = torch.full(layout.slot_index.shape, d, dtype=torch.long, device=inp.device)
        o = model._tul_core(x, x0, bigram, layout, input_ids=inp, slot_depths=table)
        got = o[2][layout.slot_valid]
        if not torch.equal(got, torch.full_like(got, d)):
            raise RuntimeError(f"the loop did not run every valid slot at depth {d}")
        stages[f"pass{d}"] = o[1]

    # ── sX: the checkpoint's OWN eval depth, no forced table ───────────────────
    o = model._tul_core(x, x0, bigram, layout, input_ids=inp)
    h_exit = o[1]
    stages["sX_exit"] = h_exit
    if check_exit:
        # The forced-depth read of the trajectory is only sound if a forced run at the
        # model's own depth IS the unforced run. Checked every batch, bit-exact, rather
        # than argued from the code.
        v = layout.slot_valid
        if not torch.equal(stages[f"pass{depth}"][v], h_exit[v]):
            raise RuntimeError(
                f"pass{depth} (forced) != the unforced exit: the forced-depth table is "
                "not reading the trajectory this model runs")

    # ── sW: what the coda reads at the prefix cells ────────────────────────────
    values, _pos = model.tul.prefix_project(h_exit, layout, layout.l_total)
    stages["sW_write"] = values

    k = int(model.cfg.tul.prefix_k)
    valid = {n: (layout.slot_valid.repeat_interleave(k, dim=1) if n == "sW_write"
                 else layout.slot_valid) for n in stages}

    packed: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in stages.items():
        raw, ro = views(model, t)
        if t.dim() == 4 and name == "sX_exit":
            # The readout view IS the model's own `_readout` — the map `val/slot_eff_rank`
            # is computed through. Checked on the headline stage, every batch.
            if not torch.equal(ro, model._readout(t)):
                raise RuntimeError("readout_view != MORPHTransformer._readout")
        packed[name] = {"raw": raw.float().cpu(), "readout": ro.float().cpu()}
    return packed, {k_: v.cpu() for k_, v in valid.items()}


# ── the run ──────────────────────────────────────────────────────────────────

def analyse(per_row: dict, order: list[str]) -> dict:
    """``{stage: {view: {...}}}`` from ``{stage: {view: [ [n,C] per row ]}}``."""
    out: dict = {}
    for si, stage in enumerate(order):
        out[stage] = {}
        for view in ("raw", "readout"):
            rows = per_row[stage][view]
            series: dict[str, list[float]] = {}
            for r in rows:
                for k, v in rank_stats(r).items():
                    series.setdefault(k, []).append(float(v))
            if si > 0:
                prev_rows = per_row[order[si - 1]][view]
                series["resid_prev"] = [residual_fraction(p, c)
                                        for p, c in zip(prev_rows, rows)]
            pooled = rank_stats(torch.cat(rows, dim=0))
            out[stage][view] = {"per_row": {k: quantiles(v) for k, v in series.items()},
                                "pooled": pooled}
    return out


COLS = [("eff_rank_raw", "rankRaw"), ("eff_rank_centered", "rankCen"),
        ("eff_rank_unit", "rankUnit"), ("pc1_frac_raw", "pc1Raw"),
        ("pc1_frac_centered", "pc1Cen"), ("cos_raw", "cosRaw"),
        ("cos_centered", "cosCen"), ("norm_mean", "|s|"),
        ("row_mean_norm_ratio", "|mu|/|s|"), ("resid_prev", "residPrev")]


def table(entry: dict, order: list[str], view: str) -> str:
    lines = [f"  [{view}] per-row median (IQR q25..q75) over {entry['rows']} rows",
             "  " + f"{'stage':<10}" + "".join(f"{h:>22}" for _k, h in COLS)]
    for stage in order:
        cells = []
        for key, _h in COLS:
            q = entry["stages"][stage][view]["per_row"].get(key)
            cells.append("-" if not q else
                         f"{q['median']:.3f}({q['q25']:.3f}..{q['q75']:.3f})")
        lines.append("  " + f"{stage:<10}" + "".join(f"{c:>22}" for c in cells))
    lines.append("")
    lines.append(f"  [{view}] POOLED over every valid slot of every row")
    lines.append("  " + f"{'stage':<10}" + "".join(f"{h:>12}" for _k, h in COLS[:-1])
                 + f"{'n':>10}")
    for stage in order:
        p = entry["stages"][stage][view]["pooled"]
        lines.append("  " + f"{stage:<10}"
                     + "".join(f"{p.get(k, float('nan')):>12.4f}" for k, _h in COLS[:-1])
                     + f"{p.get('n', 0):>10d}")
    return "\n".join(lines)


def render(out: dict) -> str:
    lines = []
    for label, entry in out.items():
        lines.append(f"=== {label}  config={entry['config']}  step={entry['step']}  "
                     f"depth={entry['depth']}  rows={entry['rows']}  "
                     f"slots={entry['valid_slots']}  d={entry['d_model']}")
        lines.append(f"  trainer reproduction: the model's OWN tul_slot_state_probe over "
                     f"these rows gives slot_eff_rank "
                     f"{entry['model_probe']['slot_eff_rank']:.4f}, slot_pairwise_cos "
                     f"{entry['model_probe']['slot_pairwise_cos']:.4f}; this instrument's "
                     f"sX_exit readout, pooled per batch, gives "
                     f"{entry['repro']['sX_exit_readout_per_batch_eff_rank']:.4f} / "
                     f"{entry['repro']['sX_exit_readout_per_batch_cos']:.4f}")
        lines.append(f"  wall {entry['wall_s']:.1f}s")
        for view in ("raw", "readout"):
            lines.append("")
            lines.append(table(entry, entry["order"], view))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=0,
                    help="forced slot depth; 0 = the checkpoint's own eval depth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from _rows import pack_rows, stream_from_loader
    from tul_samples import load_ckpt

    out: dict = {}
    for triple in a.ckpt:
        label, config, path = triple.split("=", 2)
        t0 = time.time()
        cfg = build_cfg(config, ["model.use_kernels=false"])
        tul_rt = build_tul_runtime(cfg)
        if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
            raise SystemExit(f"{label}: the rank anatomy needs a SLOT-LOOP model "
                             "(tul.tokens_through_core false)")
        model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
        model.eval()
        assert not model.training
        tc = model.cfg.tul
        own = int(tc.slot_depth_fixed) or int(tc.slot_mean_depth) or int(model.cfg.mean_depth)
        depth = a.depth or own
        order = (["s0_seed", "s1_entry"] + [f"pass{d}" for d in range(1, depth + 1)]
                 + ["sX_exit", "sW_write"])

        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
        stream = stream_from_loader(loader, a.rows * row_tokens)
        batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:-(-a.rows // a.batch)]

        per_row: dict = {s: {"raw": [], "readout": []} for s in order}
        probe_acc: dict[str, list[float]] = {}
        repro_rank, repro_cos = [], []
        n_rows = 0
        for i, (inp, _labels, layout, _idx) in enumerate(batches):
            inp, layout = inp.to(a.device), layout.to(a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                packed, valid = capture_batch(model, inp, layout, depth,
                                              check_exit=depth == own)
                for k, v in model.tul_slot_state_probe(inp, layout).items():
                    probe_acc.setdefault(k, []).append(float(v))
            for stage in order:
                m = valid[stage]
                for b in range(m.shape[0]):
                    sel = m[b]
                    for view in ("raw", "readout"):
                        per_row[stage][view].append(packed[stage][view][b][sel])
            # the trainer's pooling granularity: one batch's slots, pooled, then averaged
            ex = packed["sX_exit"]["readout"]
            vx = valid["sX_exit"]
            pooled_batch = ex[vx]
            st = rank_stats(pooled_batch)
            repro_rank.append(st["eff_rank_centered"])
            repro_cos.append(st["cos_raw"])
            n_rows += int(inp.shape[0])
            if i % 10 == 0:
                print(f"{label} batch {i}/{len(batches)} rows={n_rows} "
                      f"sX_exit readout eff_rank(centered, this batch)="
                      f"{st['eff_rank_centered']:.3f} cos={st['cos_raw']:.3f}", flush=True)

        stages = analyse(per_row, order)
        wall = time.time() - t0
        out[label] = {
            "config": config, "ckpt": path, "step": step, "depth": depth,
            "own_eval_depth": own,
            "rows": n_rows, "batch": a.batch, "order": order,
            "d_model": int(model.cfg.d_model), "prefix_k": int(tc.prefix_k),
            "valid_slots": int(sum(r.shape[0] for r in per_row["sX_exit"]["raw"])),
            "model_probe": {k: sum(v) / len(v) for k, v in probe_acc.items()},
            "repro": {
                "sX_exit_readout_per_batch_eff_rank": sum(repro_rank) / len(repro_rank),
                "sX_exit_readout_per_batch_cos": sum(repro_cos) / len(repro_cos),
            },
            "stages": stages, "wall_s": wall,
        }
        print(render({label: out[label]}), flush=True)
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w") as f:
        f.write(render(out) + "\n")
    print("wrote", a.out, "and", txt, flush=True)


if __name__ == "__main__":
    main()
