"""FLOP counting for the Parcae testbed arms (ported from the speed harness of 2026-10-14).

FlopCounterMode sees every matmul but not FlexAttention's Triton kernel, and counts flash dense. So the
matmul FLOPs come from FlopCounterMode (outside attention) and the attention FLOPs are analytic, executed
FLOPs: flash causal = 4*B*H*dh*T(T+1)/2, flex = dense * block-mask density; a forward under grad also owes a
backward of 2.5x the forward. Run on an EAGER model (FlopCounterMode cannot see inside torch.compile).
`install_attn_tally` patches lxtul.attention and the flash entry point for the life of the process, so a
process that counts FLOPs must not also be timed (bench_all.py runs the count in its own subprocess).
"""
from __future__ import annotations

import collections
import re

import torch


class AttnTally:
    """Analytic attention FLOPs (FlopCounterMode cannot see flex's Triton kernel, and counts flash
    dense). Executed FLOPs: flash causal = 4*B*H*dh*T(T+1)/2; flex = dense * block-mask density.
    A forward under grad also owes a backward of 2.5x the forward; checkpoint's first forward
    (no grad) owes none, its recompute (grad on) does."""

    def __init__(self):
        self.region = ["other"]
        self.mm = lambda: 0.0                      # set per step: matmul (non-sdpa) FLOPs so far
        self.phase = "fwd"
        self.stack = []
        self.acc = collections.defaultdict(lambda: collections.defaultdict(float))
        self.fwd = collections.defaultdict(float)
        self.bwd = collections.defaultdict(float)
        self.calls = collections.Counter()
        self._dens = {}

    def add(self, kind, B, H, Lq, Lk, dh, density, needs_bwd):
        f = 4.0 * B * H * Lq * Lk * dh * density
        r = self.region[-1]
        self.fwd[r] += f
        self.calls[(r, kind)] += 1
        if needs_bwd:
            self.bwd[r] += 2.5 * f

    def density(self, bm):
        k = id(bm)
        if k not in self._dens:
            self._dens[k] = (bm, 1.0 - float(bm.sparsity()) / 100.0)
        return self._dens[k][1]


def install_attn_tally(model):
    import lxtul.attention as A
    from parcae_lm.modules import mixer
    tally = AttnTally()
    orig_flex = A._flex

    from torch.utils._python_dispatch import _disable_current_modes

    def flex(q, k, v, block_mask=None, enable_gqa=False):
        B, H, Lq, dh = q.shape
        need = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
        tally.add("flex", B, H, Lq, k.shape[2], dh, tally.density(block_mask), need)
        with _disable_current_modes():     # compiled flex cannot trace under FlopCounterMode
            return orig_flex(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)
    A._flex = flex
    orig_mask = A._mask

    def mask(*a, **k):
        with _disable_current_modes():
            return orig_mask(*a, **k)
    A._mask = mask

    ns = mixer.flash_attn
    orig_fa = ns.flash_attn_func

    def fa(q, k, v, causal=False, window_size=(-1, -1)):
        B, T, H, dh = q.shape
        need = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
        pairs = T * (T + 1) / 2 if causal else T * T
        tally.add("flash", B, H, T, T, dh, pairs / (T * T), need)
        return orig_fa(q, k, v, causal=causal, window_size=window_size)
    ns.flash_attn_func = fa

    pat = re.compile(r"(twin_prelude|sd_blocks|transformer\.prelude|core_block|transformer\.coda|transformer\.h)\.\d+$")
    for name, mod in model.named_modules():
        m = pat.search(name)
        if not m:
            continue
        reg = {"twin_prelude": "ema_twin_prelude", "sd_blocks": "span_decoder", "transformer.prelude": "prelude",
               "core_block": "core", "transformer.coda": "coda", "transformer.h": "gpt_blocks"}[m.group(1)]
        def pre(mod_, args, reg=reg):
            tally.region.append(reg)
            tally.stack.append(tally.mm())

        def post(mod_, args, out, reg=reg):
            tally.region.pop()
            d = tally.mm() - tally.stack.pop()
            a = tally.acc[reg]
            a["all"] += d
            if torch.is_grad_enabled() and isinstance(out, torch.Tensor) and out.requires_grad:
                a["grad"] += d
            if tally.phase == "bwd":
                a["recompute"] += d
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
    return tally


SDPA_RX = re.compile(r"scaled_dot_product|flash_attention|efficient_attention")


def mm_total(fc) -> float:
    g = fc.flop_counts.get("Global", {})
    return float(sum(f for op, f in g.items() if not SDPA_RX.search(str(op))))


def muon_flops(opt) -> float:
    tot = 0.0
    for g in opt.param_groups:
        if g["kind"] != "muon":
            continue
        for p in g["params"]:
            r, c = p.shape[-2:]
            a, b = min(r, c), max(r, c)
            tot += 5 * (4 * a * a * b + 2 * a ** 3)
    return tot


def spiral_layer_equivalents(levels, L: int, n_prelude: int, n_core: int, n_coda: int, span_groups: float,
                             span_depth: float) -> float:
    """Linear-layer cost per token of a spiral arm (lxtul/spiral.py), in layers run on all L positions:
    prelude + coda, plus every level's core passes on its group sequence. A chunk level runs `passes` (depth poisson: `span_depth`) x
    `n_core` layers on L // size positions; the span level runs `span_depth` x `n_core` layers on
    `span_groups` positions (pass the real mean span count for the cost the data owes, `max_spans` for
    the static shape the GPU runs). Attention is extra (quadratic in the group length); FlopCounterMode
    plus install_attn_tally count both exactly, this is the estimate the prereg quotes."""
    eq = float(n_prelude + n_coda)
    for lv in levels:
        if lv.kind == "chunk" and lv.depth == "poisson":
            eq += span_depth * n_core * (L // lv.size) / L        # a poisson chunk level loops at the sampled depth
        elif lv.kind == "chunk":
            eq += lv.passes * n_core * (L // lv.size) / L
        else:
            eq += span_depth * n_core * span_groups / L
    return eq
