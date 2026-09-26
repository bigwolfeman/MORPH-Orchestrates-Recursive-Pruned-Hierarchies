"""LX-Concept groundwork: the ONE home of the span join, the teacher reads, the student
reads, the two dictionaries, the CoCoMix attribution and the pre-check head.

Design context (prereg ``lab/experiments/planned/2026-09-26-lx-concept-precheck.md``):
LX-Concept would make each LX rollout carry a discrete per-span CONCEPT hypothesis predicted
from context (CoCoMix, arXiv 2502.08524). The concept labels come from a TEACHER, the LCTUL
VAE-stage span autoencoder (``tul_code_vae``, ``checkpoints/morph/tul-code-vae``): E pools a
span's code from that span's own prelude states and the frozen coda decodes the span from it
(docs/tul-code-spec.md §3.3, §17). The scripts that use this module:

  concept_audit.py     step 1: tokenizer / boundary-rule alignment, oracle CE vs fp01's coda
  concept_dump.py      step 2: E's codes and fp01's entry / exit states, keyed by span
  concept_dict.py      step 3: k-means span types, a TopK SAE, CoCoMix attribution labels
  concept_precheck.py  step 4: can fp01's slot state predict the next span's concept?

TERMS (one meaning each):

  slot s         a valid slot of a packed row; it closes span s (``bag_id == s``) and
                 precedes span s+1 (``bag_id == s+1``).
  next span      span s+1 of slot s: the token positions with ``bag_id == s+1``. E's code at
                 slot s is the code of this span (``TULCodeEncoder``); fp01's parallel head
                 at slot s targets the same span (``span_slots``, offset 1).
  coded slot     a slot with ``code_target_valid``: s and s+1 are both valid, so its next
                 span is complete. The only slots this module keys.
  span key       ``(first, last)``: the stream indices of the next span's first and last
                 token. Two packings of the same stream agree on a span iff the keys match;
                 every join in the pipeline is on this key, never on (row, slot).
  code           E's output at a coded slot, ``[M = 2, C = 1024]``, unit RMS per cell,
                 flattened to 2,048 for the dictionaries.
  span positions the token positions of the next span whose label counts in the coda CE
                 (``_tul_half_weights`` > 0 and a label): the positions that read the cells
                 of slots <= s under the strict geometry and predict span s+1's tokens 2..n
                 and span s+2's first token. E pools exactly these positions' inputs.
  span NLL       the sum of the coda's per-position CE over the span positions.
  entry          the slot's loop ENTRY state: ``input_norm(prelude)`` gathered at the slot
                 position (``_tul_core``'s ``e``), mean over the Hyper-Connection streams.
                 Under the strict geometry it has seen span s alone.
  exit           ``_readout`` of the slot's exit state (the tensor the parallel head reads,
                 captured at ``_tul_spandec_par_loss``), mean over the K = 4 rollouts.
  exit_raw       the same exit state, stream mean, rollout mean, before ``_readout``.

The teacher runs on its OWN checkpoint's prelude: E never sees a live model's states (the
2026-09-18 lesson, memory "frozen encoder on a live front is not a fixed target").
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

TEACHER_CONFIG = "tul_code_vae"
TEACHER_CKPT = "/home/wolfe/morph-to/checkpoints/morph/tul-code-vae/step_10000.pt"
STUDENT_CONFIG = "tul_slot_spandec_strict_e4probe_fp01"
STUDENT_DIR = "/home/wolfe/morph-to/checkpoints/morph/lxtul-e4probe-fp01"
SCRATCH = "/home/wolfe/morph-scratch/concept"
# Documents skipped before the TRAIN region's rows. The probes' "validation" rows are the
# OWT stream from document 0 (``create_dataloader`` falls back to the train split for OWT),
# which the trainer also reads first, at steps 0..80 of every run. 10k steps x 6 rows x
# ~1.1k tokens is ~66M tokens, about 60k documents; 200k documents is past what fp01 and
# the VAE teacher ever read.
TRAIN_SKIP_DOCS = 200_000


class Stop(Exception):
    """Raised by a stub seam to end a forward early; the captured tensors keep their graph."""


class Spy:
    """Wrap ``obj.<name>`` for the ``with`` block: record every call's args (and outputs),
    call through, or with ``stop=True`` record the args and raise :class:`Stop` instead.
    Restores the attribute on exit (the instance attribute is deleted)."""

    def __init__(self, obj, name: str, stop: bool = False):
        self.obj, self.name, self.stop = obj, name, stop
        self.calls: list[tuple] = []
        self.outs: list = []

    def __enter__(self):
        if self.name in vars(self.obj):
            raise RuntimeError(f"{self.name} is already wrapped on this object")
        real = getattr(self.obj, self.name)

        def wrap(*a, **k):
            self.calls.append((a, k))
            if self.stop:
                raise Stop(self.name)
            r = real(*a, **k)
            self.outs.append(r)
            return r
        setattr(self.obj, self.name, wrap)
        return self

    def __exit__(self, *exc):
        delattr(self.obj, self.name)
        return False

    def one(self) -> tuple:
        if len(self.calls) != 1:
            raise RuntimeError(f"seam {self.name} reached {len(self.calls)} times, expected 1")
        return self.calls[0]


# ── span keys and the join ──────────────────────────────────────────────────────────


def coded_slots(layout) -> torch.Tensor:
    """``[B, S]`` bool: the coded slots (``code_target_valid``, the teacher's own rule)."""
    from morph.model.tul_code import code_target_valid
    return code_target_valid(layout)


def span_keys(layout, idx: torch.Tensor) -> dict[str, np.ndarray]:
    """One record per coded slot of the batch, in (row, slot) order.

    ``idx`` ``[B, L]`` is the stream index of the input token at each position (-1 at slot
    and pad positions), as ``_rows.pack_rows`` returns it. Keys: ``row``, ``slot``,
    ``first`` / ``last`` (the next span's first / last stream index), ``n_tok`` (its token
    positions). A coded slot whose next span holds no token RAISES (the packer never makes
    one: ``code_target_valid`` requires slot s+1, which closes a non-empty span)."""
    ok = coded_slots(layout).cpu()
    bag = layout.bag_id.cpu()
    tok = (~layout.slot_mask).cpu()
    idx = idx.cpu()
    rows, slots, first, last, ntok = [], [], [], [], []
    B, S = ok.shape
    for b in range(B):
        for s in ok[b].nonzero().flatten().tolist():
            sel = tok[b] & (bag[b] == s + 1)
            ix = idx[b][sel]
            if ix.numel() == 0 or bool((ix < 0).any()):
                raise RuntimeError(f"row {b} slot {s}: next span has no stream-indexed token")
            if not bool((ix[1:] - ix[:-1] == 1).all()):
                raise RuntimeError(f"row {b} slot {s}: next span is not contiguous in the stream")
            rows.append(b)
            slots.append(s)
            first.append(int(ix[0]))
            last.append(int(ix[-1]))
            ntok.append(int(ix.numel()))
    return {"row": np.asarray(rows, np.int64), "slot": np.asarray(slots, np.int64),
            "first": np.asarray(first, np.int64), "last": np.asarray(last, np.int64),
            "n_tok": np.asarray(ntok, np.int64)}


def key64(first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """One int64 per span key; injective for stream indices below 2^40 and spans < 2^20."""
    first = np.asarray(first, np.int64)
    ln = np.asarray(last, np.int64) - first
    if (first < 0).any() or (first >= 1 << 40).any() or (ln < 0).any() or (ln >= 1 << 20).any():
        raise ValueError("span key out of range")
    return (first << 20) | ln


def join_keys(ka: dict, kb: dict) -> tuple[np.ndarray, np.ndarray]:
    """Indices ``(ia, ib)`` of the spans present in both key sets, matched on
    ``(first, last)``. Duplicate keys inside one set RAISE (a stream index belongs to one
    span of one packing)."""
    a = key64(ka["first"], ka["last"])
    b = key64(kb["first"], kb["last"])
    if np.unique(a).size != a.size or np.unique(b).size != b.size:
        raise ValueError("duplicate span keys inside one packing")
    _c, ia, ib = np.intersect1d(a, b, assume_unique=True, return_indices=True)
    return ia, ib


def concat_keys(parts: list[dict], row_offsets: list[int] | None = None) -> dict:
    """Concatenate per-batch key dicts; ``row`` is offset to a global row number."""
    out: dict[str, list] = {}
    off = 0
    for i, p in enumerate(parts):
        for k, v in p.items():
            out.setdefault(k, []).append(v + (off if k == "row" else 0))
        off += row_offsets[i] if row_offsets is not None else 0
    return {k: np.concatenate(v) for k, v in out.items()}


def compare_packing(batches_a: list, batches_b: list) -> dict:
    """Two packings of ONE stream, compared row by row and span by span.

    ``batches_*`` are ``_rows.pack_rows`` outputs. Row-level: the input ids, labels,
    ``slot_mask``, ``bag_id``, ``slot_index`` and ``slot_valid`` of every row; span-level:
    every coded slot's next-span key and token ids, joined on the key. Returns the counts;
    identical packings read zero mismatches everywhere."""
    def rows(batches):
        out = []
        for inp, lab, lay, idx in batches:
            for b in range(inp.shape[0]):
                out.append((inp[b], lab[b], lay.slot_mask[b], lay.bag_id[b],
                            lay.slot_index[b], lay.slot_valid[b], idx[b]))
        return out

    ra, rb = rows(batches_a), rows(batches_b)
    n = min(len(ra), len(rb))
    fields = ["input_ids", "labels", "slot_mask", "bag_id", "slot_index", "slot_valid",
              "stream_index"]
    row_mis = {f: 0 for f in fields}
    for i in range(n):
        for f, x, y in zip(fields, ra[i], rb[i]):
            if x.shape != y.shape or not torch.equal(x, y):
                row_mis[f] += 1

    def span_tokens(batches):
        keys, toks = [], {}
        for inp, _lab, lay, idx in batches:
            k = span_keys(lay, idx)
            for j in range(k["row"].size):
                b, s = int(k["row"][j]), int(k["slot"][j])
                sel = (~lay.slot_mask[b]) & (lay.bag_id[b] == s + 1)
                toks[(int(k["first"][j]), int(k["last"][j]))] = inp[b][sel].tolist()
            keys.append(k)
        return concat_keys(keys), toks

    ka, ta = span_tokens(batches_a)
    kb, tb = span_tokens(batches_b)
    ia, ib = join_keys(ka, kb)
    tok_mis = sum(1 for j in ia if ta[(int(ka["first"][j]), int(ka["last"][j]))]
                  != tb[(int(ka["first"][j]), int(ka["last"][j]))])
    return {"rows_a": len(ra), "rows_b": len(rb), "rows_compared": n,
            "row_field_mismatches": row_mis,
            "spans_a": int(ka["row"].size), "spans_b": int(kb["row"].size),
            "spans_joined": int(ia.size),
            "spans_only_a": int(ka["row"].size - ia.size),
            "spans_only_b": int(kb["row"].size - ib.size),
            "span_token_mismatches": int(tok_mis)}


def region_batches(cfg, rt, region: str, rows: int, batch: int, skip_docs: int):
    """Packed rows of one region, in ``val_batches`` format ``(inp, labels, layout, idx)``.

    ``val``: the first ``rows`` rows of the validation stream (``val_batches``, the rows
    ``core_depth_sweep.py`` and the LX scorers read). ``train``: the OWT stream after
    ``skip_docs`` documents, packed by the same packer. Stream indices of the train region
    are offset by 10^12 so no train key can equal a val key."""
    from _rows import pack_rows, stream_from_loader
    from lxtul_e_stage1_score import val_batches

    from morph.training.data import create_dataloader
    if region == "val":
        return val_batches(cfg, rt, rows, batch)[0]
    if region != "train":
        raise ValueError(f"region must be val|train, got {region!r}")
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8, split="train",
                               skip_samples=int(skip_docs), bag_size=0, tul=None)
    row_tokens = rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, rows * row_tokens)
    got = pack_rows(stream, rt, cfg, batch, False)[:-(-rows // batch)]
    if sum(b[0].shape[0] for b in got) < rows:
        raise SystemExit(f"train region packed fewer than {rows} rows")
    off = 10 ** 12
    return [(i, lab, lay, torch.where(ix >= 0, ix + off, ix)) for i, lab, lay, ix in got]


# ── the teacher: codes, per-position CE, per-span gradients ─────────────────────────


def _autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")


def scored_mask(m, labels: torch.Tensor, layout) -> torch.Tensor:
    """``[B, L]`` bool: the positions whose label counts in the coda CE (weight > 0)."""
    row_w, _p, _z = m._tul_half_weights(labels, layout)
    return (row_w.view(labels.shape) * (labels != -100)).bool()


def noisy_codes(z: torch.Tensor, ok: torch.Tensor, noise: float, seed: int) -> torch.Tensor:
    """The TRAINING statistic of a truth cell on a ``code_noise_renorm`` arm:
    ``rmsnorm(z + noise * eps) * ok`` with eps from a seeded CPU generator (fp32)."""
    from morph.model.tul_code import code_rmsnorm
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    eps = torch.randn(tuple(z.shape), generator=g, dtype=torch.float32).to(z.device)
    out = code_rmsnorm(z.float() + float(noise) * eps)
    return (out * ok.view(*ok.shape, 1, 1).float()).to(z.dtype)


@torch.no_grad()
def teacher_read(m, inp, labels, layout, device: str, noise: float = 0.0, seed: int = 0,
                 tol: float = 2e-3) -> dict:
    """One labelled eval forward of the teacher, ``code_mode="encoder"``: the coda reads E's
    code of every span (``noise > 0``: the code at the training statistic, seeded).

    Returns ``z`` ``[B, S, M, C]`` (E's clean code, fp32 on CPU), ``ok`` ``[B, S]``, and
    ``ce`` ``[B, L]`` (the coda's per-position CE, 0 where not scored) with ``scored``
    ``[B, L]``. Self-check: the scored mean equals the forward's own ``ce_tokens`` within
    ``tol`` (RAISES), so these are the numbers the trainer's val pass reads."""
    from morph.model.fused_ce import fused_linear_label_logprob
    enc = m.tul_code_enc
    if enc is None:
        raise RuntimeError("teacher has no tul_code_enc: not a tul.code model")
    got: dict = {}
    real = enc.forward

    def enc_fwd(xs, lay):
        z, ok = real(xs, lay)
        got["z"], got["ok"] = z, ok
        if noise > 0.0:
            z = noisy_codes(z, ok, noise, seed)
        return z, ok

    enc.forward = enc_fwd
    try:
        with _autocast(device), Spy(m, "_tul_group_losses") as sp:
            out = m(inp.to(device), labels=labels.to(device), slot_layout=layout.to(device),
                    code_mode="encoder")
    finally:
        del enc.forward
    xh = sp.one()[0][0]
    lab = labels.to(device)
    with _autocast(device):
        lp = fused_linear_label_logprob(xh.reshape(-1, xh.shape[-1]), m.embed.lm_weight(),
                                        lab.reshape(-1), chunk_size=m.cfg.ce_chunk_size,
                                        mask_token_id=m.cfg.tul.slot_id).view(lab.shape).float()
    sc = scored_mask(m, lab, layout.to(device))
    ce = (-lp) * sc
    dev = abs(float(ce.sum() / sc.sum()) - float(out["ce_tokens"]))
    if dev > tol:
        raise RuntimeError(f"teacher offline CE differs from the forward's ce_tokens by {dev}")
    return {"z": got["z"].float().cpu(), "ok": got["ok"].cpu(), "ce": ce.cpu(),
            "scored": sc.cpu(), "dev": dev}


def span_nll(ce: torch.Tensor, scored: torch.Tensor, layout, keys: dict) -> np.ndarray:
    """Span NLL (sum over span positions) of every keyed span, in key order."""
    out = np.zeros(keys["row"].size, dtype=np.float64)
    bag = layout.bag_id.cpu()
    tok = (~layout.slot_mask).cpu()
    for j in range(keys["row"].size):
        b, s = int(keys["row"][j]), int(keys["slot"][j])
        sel = tok[b] & (bag[b] == s + 1) & scored[b]
        out[j] = float(ce[b][sel].double().sum())
    return out


def _take_layout(layout, rows: torch.Tensor):
    from morph.model.tul_layout import SlotLayout
    return SlotLayout(slot_mask=layout.slot_mask[rows], bag_id=layout.bag_id[rows],
                      slot_index=layout.slot_index[rows], slot_valid=layout.slot_valid[rows],
                      prefix_k=layout.prefix_k)


def span_grads(m, inp, labels, layout, rows: np.ndarray, slots: np.ndarray,
               cells: torch.Tensor, device: str, group: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """EXACT per-span gradient of the teacher's span NLL w.r.t. the cells it reads.

    For each requested span j (row ``rows[j]``, slot ``slots[j]``), the teacher's coda reads
    E's clean code at every slot except slot ``slots[j]``, which holds
    ``code_rmsnorm(cells[j])`` (``cells`` ``[n, M, C]``, e.g. an SAE reconstruction D(c));
    the loss is span j's NLL alone. Spans are run as independent COPIES of their row, ``group``
    copies per forward, so a copy's gradient is its own span's and nothing else's (under
    the strict geometry span s+2.. also read cell s; a summed loss would mix them in).
    The forward stops at the coda output (``_tul_group_losses`` is a stub), and only the
    span positions go through the LM head.

    Returns ``(g [n, M*C] fp32, nll [n] fp64)``: d NLL_j / d cells[j] and NLL_j itself."""
    from morph.model.fused_ce import fused_linear_label_logprob
    from morph.model.tul_code import code_rmsnorm
    enc = m.tul_code_enc
    n = len(rows)
    M, C = int(cells.shape[1]), int(cells.shape[2])
    g_out = np.zeros((n, M * C), dtype=np.float32)
    nll_out = np.zeros(n, dtype=np.float64)
    lay_d = layout.to(device)
    inp_d, lab_d = inp.to(device), labels.to(device)
    sc_all = scored_mask(m, lab_d, lay_d)
    real = enc.forward
    for c0 in range(0, n, group):
        ix = np.arange(c0, min(n, c0 + group))
        r = torch.as_tensor(rows[ix], device=device)
        s = torch.as_tensor(slots[ix], device=device)
        G = r.numel()
        leaf = cells[ix].to(device=device, dtype=torch.float32).clone().requires_grad_(True)
        lay_g = _take_layout(lay_d, r)
        S = lay_g.slot_index.shape[1]
        sel_slot = F.one_hot(s, S).bool()                                   # [G, S]

        def enc_fwd(xs, lay, leaf=leaf, sel_slot=sel_slot):
            z, ok = real(xs, lay)
            cell = code_rmsnorm(leaf).to(z.dtype)                           # [G, M, C]
            z = torch.where(sel_slot.view(G, S, 1, 1), cell.unsqueeze(1), z)
            return z, ok

        enc.forward = enc_fwd
        try:
            with torch.enable_grad(), _autocast(device), \
                    Spy(m, "_tul_group_losses", stop=True) as sp:
                try:
                    m(inp_d[r], labels=lab_d[r], slot_layout=lay_g, code_mode="encoder")
                except Stop:
                    pass
        finally:
            del enc.forward
        xh = sp.one()[0][0]                                                 # [G, L, C]
        pos = ((~lay_g.slot_mask) & (lay_g.bag_id == (s + 1).view(G, 1)) & sc_all[r])
        gi, pi = pos.nonzero(as_tuple=True)
        with torch.enable_grad(), _autocast(device):
            lp = fused_linear_label_logprob(xh[gi, pi], m.embed.lm_weight(), lab_d[r][gi, pi],
                                            chunk_size=m.cfg.ce_chunk_size,
                                            mask_token_id=m.cfg.tul.slot_id).float()
            per = torch.zeros(G, device=device, dtype=torch.float32).index_add(0, gi, -lp)
            (grad,) = torch.autograd.grad(per.sum(), [leaf])
        g_out[ix] = grad.reshape(G, M * C).float().cpu().numpy()
        nll_out[ix] = per.detach().double().cpu().numpy()
    return g_out, nll_out


# ── the student: entry and exit states ──────────────────────────────────────────────


@torch.no_grad()
def student_states(m, inp, labels, layout, device: str, depth: int = 6) -> dict:
    """fp01's slot states at forced depth ``depth`` (every valid slot), rollout mean.

    ``entry`` from ``_tul_core``'s own input (``gather_valid(input_norm(x))`` at the slot
    positions, stream mean); ``exit`` / ``exit_raw`` from the ``h_slots`` the parallel head
    reads (``_tul_spandec_par_loss``, where the forward is stopped: no coda runs). All
    ``[B, S, C]`` fp32 on CPU."""
    from morph.model.tul import gather_valid
    if m.tul_spandec_par is None:
        raise RuntimeError("student has no parallel head: the exit seam does not exist")
    K = max(int(getattr(m, "_code_enum_k", 0)), 1)
    table = torch.full(layout.slot_index.shape, int(depth), dtype=torch.long)
    lay = layout.to(device)
    with _autocast(device), Spy(m, "_tul_core") as core, \
            Spy(m, "_tul_spandec_par_loss", stop=True) as head:
        try:
            m(inp.to(device), labels=labels.to(device), slot_layout=lay, slot_depths=table)
        except Stop:
            pass
    x = core.one()[0][0]                                                 # [B, L, (n,) C]
    B = inp.shape[0]
    if x.shape[0] != B:
        raise RuntimeError(f"_tul_core got {x.shape[0]} rows, expected the base batch {B}")
    h_slots = head.one()[0][0]                                           # [K*B, S, (n,) C]
    with _autocast(device):
        xn = m.input_norm(x)
        e = gather_valid(xn, lay.slot_index, lay.slot_valid)
        r = m._readout(h_slots)
    hc = h_slots.dim() == 4
    entry = (e.float().mean(2) if hc else e.float())
    raw = (h_slots.float().mean(2) if hc else h_slots.float())
    S, C = raw.shape[1], raw.shape[-1]
    return {"entry": entry.cpu(),
            "exit": r.float().view(K, B, S, C).mean(0).cpu(),
            "exit_raw": raw.view(K, B, S, C).mean(0).cpu()}


# ── dictionaries ────────────────────────────────────────────────────────────────────


def kmeans(X: torch.Tensor, M: int, iters: int = 50, seed: int = 0, chunk: int = 65536
           ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Lloyd's k-means with k-means++ init on the rows of ``X`` (fp32, any device).

    Returns ``(centroids [M, D], labels [N], inertia per iteration)``. An empty cluster is
    re-seeded at the point farthest from its centroid (so M clusters always exist)."""
    X = X.float()
    N = X.shape[0]
    if N < M:
        raise ValueError(f"k-means needs at least M = {M} points, got {N}")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x2 = (X * X).sum(1)
    cent = torch.empty(M, X.shape[1], device=X.device)
    i0 = int(torch.randint(N, (1,), generator=g))
    cent[0] = X[i0]
    d2 = (x2 - 2 * X @ cent[0] + cent[0].pow(2).sum()).clamp_min(0)
    for j in range(1, M):
        p = (d2 / d2.sum()).double().cpu()
        i = int(torch.multinomial(p, 1, generator=g))
        cent[j] = X[i]
        d2 = torch.minimum(d2, (x2 - 2 * X @ cent[j] + cent[j].pow(2).sum()).clamp_min(0))
    inertia: list[float] = []
    lab = torch.zeros(N, dtype=torch.long, device=X.device)
    for _ in range(iters):
        lab, dmin = assign(X, cent, chunk)
        inertia.append(float(dmin.sum()))
        cnt = torch.bincount(lab, minlength=M).float()
        new = torch.zeros_like(cent).index_add_(0, lab, X)
        empty = cnt == 0
        if bool(empty.any()):
            far = torch.topk(dmin, int(empty.sum())).indices
            new[empty] = X[far]
            cnt[empty] = 1.0
        new = new / cnt.unsqueeze(1)
        if torch.allclose(new, cent):
            cent = new
            break
        cent = new
    lab, dmin = assign(X, cent, chunk)
    inertia.append(float(dmin.sum()))
    return cent, lab, inertia


def assign(X: torch.Tensor, cent: torch.Tensor, chunk: int = 65536
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest centroid (squared Euclidean) of every row, and that squared distance."""
    c2 = (cent * cent).sum(1)
    labs, ds = [], []
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk].float()
        d = (x * x).sum(1, keepdim=True) - 2 * x @ cent.T + c2
        v, a = d.min(1)
        labs.append(a)
        ds.append(v.clamp_min(0))
    return torch.cat(labs), torch.cat(ds)


class TopKSAE(torch.nn.Module):
    """TopK sparse autoencoder (Gao et al. 2024, the SAE CoCoMix uses).

    ``pre(x) = W_enc (x - b_dec) + b_enc`` (all ``n`` latents: CoCoMix's c^pre),
    ``c = TopK_k(pre)`` (the k largest kept, the rest 0; ReLU on the kept values),
    ``D(c) = c W_dec + b_dec`` with unit-norm decoder rows."""

    def __init__(self, d_in: int, n_latents: int, k: int, seed: int = 0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        w = torch.randn(n_latents, d_in, generator=g)
        w = w / w.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(w)                          # [n, D]
        self.W_enc = torch.nn.Parameter(w.T.clone())                # [D, n]
        self.b_enc = torch.nn.Parameter(torch.zeros(n_latents))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
        self.k = int(k)

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(c dense [N, n] with k nonzeros per row, pre [N, n])``."""
        p = self.pre(x)
        v, i = torch.topk(p, self.k, dim=-1)
        c = torch.zeros_like(p).scatter(-1, i, F.relu(v))
        return c, p

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec + self.b_dec

    @torch.no_grad()
    def renorm_(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)


def train_sae(X: torch.Tensor, n_latents: int, k: int, steps: int, batch: int = 1024,
              lr: float = 3e-4, aux_k: int = 256, aux_coef: float = 1 / 32,
              dead_after: int = 2_000_000, seed: int = 0, log_every: int = 0
              ) -> tuple[TopKSAE, dict]:
    """Fit a :class:`TopKSAE` on the rows of ``X`` (device = X's) with Adam, the AuxK term
    for latents dead for ``dead_after`` samples, and unit-norm decoder rows after each step.
    ``b_dec`` starts at the data mean. Returns the model and its training stats."""
    torch.manual_seed(int(seed))
    dev = X.device
    D = X.shape[1]
    sae = TopKSAE(D, n_latents, k, seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(X.float().mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    var = float(X.float().var(0).sum())
    since = torch.zeros(n_latents, device=dev)
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    hist = []
    for step in range(steps):
        ix = torch.randint(X.shape[0], (batch,), generator=g).to(dev)
        x = X[ix].float()
        c, p = sae.encode(x)
        xh = sae.decode(c)
        err = x - xh
        mse = err.pow(2).sum(1).mean()
        loss = mse
        fired = (c > 0).any(0)
        since = torch.where(fired, torch.zeros_like(since), since + batch)
        dead = since > dead_after
        if aux_k > 0 and bool(dead.any()):
            pd = p.masked_fill(~dead, float("-inf"))
            ka = min(aux_k, int(dead.sum()))
            v, i = torch.topk(pd, ka, dim=-1)
            ca = torch.zeros_like(p).scatter(-1, i, F.relu(v))
            ea = ca @ sae.W_dec
            aux = (err.detach() - ea).pow(2).sum(1).mean() / err.detach().pow(2).sum(1).mean().clamp_min(1e-8)
            loss = loss + aux_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            # the gradient component parallel to each unit decoder row does nothing once the
            # row is renormalised; remove it (the standard TopK SAE update)
            w = sae.W_dec
            if w.grad is not None:
                w.grad -= (w.grad * w).sum(1, keepdim=True) * w
        opt.step()
        sae.renorm_()
        if log_every and (step % log_every == 0 or step == steps - 1):
            hist.append({"step": step, "fvu": float(mse.detach()) / var,
                         "dead": int(dead.sum())})
    return sae, {"steps": steps, "batch": batch, "lr": lr, "aux_k": aux_k,
                 "dead_after": dead_after, "history": hist, "data_var": var}


@torch.no_grad()
def sae_eval(sae: TopKSAE, X: torch.Tensor, chunk: int = 8192) -> dict:
    """Fraction of variance unexplained, dead-latent fraction, mean cosine on ``X``."""
    se = tot = cos = 0.0
    fired = torch.zeros(sae.W_dec.shape[0], dtype=torch.bool, device=X.device)
    mu = X.float().mean(0)
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk].float()
        c, _p = sae.encode(x)
        xh = sae.decode(c)
        se += float((x - xh).pow(2).sum())
        tot += float((x - mu).pow(2).sum())
        cos += float(F.cosine_similarity(x, xh, dim=1).sum())
        fired |= (c > 0).any(0)
    return {"fvu": se / tot, "dead_frac": float((~fired).float().mean()),
            "cos": cos / X.shape[0], "n": int(X.shape[0])}


def attribution(sae: TopKSAE, pre: torch.Tensor, g_cells: torch.Tensor) -> torch.Tensor:
    """CoCoMix's attribution score over EVERY latent (arXiv 2502.08524, Eq. for a_t):
    ``a = c^pre * d(-log p)/dc``. ``D`` is linear in ``c``, so ``d/dc = W_dec g`` with ``g``
    the gradient w.r.t. the decoded code (``span_grads``). ``pre`` ``[N, n]``,
    ``g_cells`` ``[N, D]`` -> ``[N, n]``."""
    return pre.float() * (g_cells.float() @ sae.W_dec.T.float())


def top_attr(a: torch.Tensor, k: int = 4, largest: bool = True) -> torch.Tensor:
    """The ``k`` latent indices with the highest (CoCoMix) or lowest attribution per row."""
    return torch.topk(a, k, dim=-1, largest=largest).indices


# ── the pre-check head ──────────────────────────────────────────────────────────────


def set_ce(logp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-row CoCoMix concept loss: the mean over the T target indices of ``-logp[y_t]``.
    ``logp`` ``[N, n_classes]`` log-probabilities, ``y`` ``[N, T]`` long (T = 1: plain CE)."""
    return -logp.gather(1, y).mean(1)


def marginal_logp(y_train: torch.Tensor, n_classes: int, alpha: float = 0.5) -> torch.Tensor:
    """The context-free prior: additive-``alpha`` smoothed frequencies of every target index
    in the training labels (each of a row's T indices counts once). ``[n_classes]`` log-p."""
    cnt = torch.bincount(y_train.reshape(-1).cpu(), minlength=n_classes).double() + alpha
    return (cnt / cnt.sum()).log().float()


def hits(logp: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """Per row: the fraction of the row's T targets inside the top-``k`` predicted classes
    (T = 1: top-k accuracy)."""
    top = torch.topk(logp, k, dim=1).indices                             # [N, k]
    return (y.unsqueeze(-1) == top.unsqueeze(1)).any(-1).float().mean(1)


class Head(torch.nn.Module):
    """Linear softmax head (``hidden == 0``) or one hidden layer of GELU units."""

    def __init__(self, d_in: int, n_classes: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
                                           torch.nn.Linear(hidden, n_classes))
        else:
            self.net = torch.nn.Linear(d_in, n_classes)

    def forward(self, x):
        return torch.log_softmax(self.net(x).float(), dim=-1)


def fit_head(Xtr: torch.Tensor, ytr: torch.Tensor, Xdv: torch.Tensor, ydv: torch.Tensor,
             n_classes: int, hidden: int = 0, lr: float = 1e-3, wd: float = 1e-2,
             batch: int = 1024, max_epochs: int = 40, patience: int = 3, seed: int = 0,
             prior: torch.Tensor | None = None) -> tuple[Head, dict]:
    """Fit :class:`Head` on standardised features by AdamW on :func:`set_ce`, early-stopped
    on the dev rows (the best dev epoch's weights are returned). ``prior`` (log-p) sets the
    output bias of a linear head at init, so epoch 0 starts at the marginal."""
    torch.manual_seed(int(seed))
    dev = Xtr.device
    # at least 16 steps an epoch, so a small fit set still gets hundreds of updates
    batch = max(32, min(int(batch), Xtr.shape[0] // 16))
    h = Head(Xtr.shape[1], n_classes, hidden).to(dev)
    if prior is not None:
        last = h.net if hidden == 0 else h.net[-1]
        with torch.no_grad():
            last.bias.copy_(prior.to(dev))
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 7)

    def dev_ce() -> float:
        h.eval()
        with torch.no_grad():
            v = torch.cat([set_ce(h(Xdv[i:i + 8192]), ydv[i:i + 8192])
                           for i in range(0, Xdv.shape[0], 8192)])
        h.train()
        return float(v.mean())

    best = dev_ce()
    best_state = {k: v.clone() for k, v in h.state_dict().items()}
    hist = [best]
    bad = 0
    for ep in range(max_epochs):
        perm = torch.randperm(Xtr.shape[0], generator=g).to(dev)
        for i in range(0, Xtr.shape[0], batch):
            ix = perm[i:i + batch]
            loss = set_ce(h(Xtr[ix]), ytr[ix]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = dev_ce()
        hist.append(v)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in h.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    h.load_state_dict(best_state)
    h.eval()
    return h, {"dev_ce_by_epoch": hist, "best_dev_ce": best, "epochs_run": len(hist) - 1}


def standardise(Xtr: torch.Tensor, *others: torch.Tensor) -> list[torch.Tensor]:
    """Per-feature z-score with the TRAIN mean and std (std floored at 1e-6)."""
    mu = Xtr.float().mean(0, keepdim=True)
    sd = Xtr.float().std(0, keepdim=True).clamp_min(1e-6)
    return [((x.float() - mu) / sd) for x in (Xtr, *others)]


def row_block_ci(a: np.ndarray, b: np.ndarray, rows: np.ndarray, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """``mean(a) - mean(b)`` over paired spans, 95 % bootstrap over ROWS (spans of one row
    are not independent). Rows are the resampling unit of ``_stats.paired_bootstrap_ci``."""
    from _stats import paired_bootstrap_ci
    _u, inv = np.unique(rows, return_inverse=True)
    nb = int(inv.max()) + 1
    sa = np.bincount(inv, weights=np.asarray(a, np.float64), minlength=nb)
    sb = np.bincount(inv, weights=np.asarray(b, np.float64), minlength=nb)
    c = np.bincount(inv, minlength=nb).astype(np.float64)
    return paired_bootstrap_ci(sa, sb, c, n_boot=n_boot, seed=seed)


def entropy_nats(logp: torch.Tensor) -> float:
    p = logp.double().exp()
    return float(-(p * logp.double()).sum())


def log_uniform(n: int) -> float:
    return math.log(n)
