"""``tul.vq_codes`` — the DISCRETE thought: K codes per span instead of one vector.

THE MEASURED DEFECT. A row's written slot states sit at effective rank **5.7598** in 1024
dimensions with mean pairwise cosine **0.7104** (``val/slot_eff_rank`` /
``val/slot_pairwise_cos`` on ``slot-spandec-strict``), 5.7-7.3 / 0.72-0.77 across the slot
family, 1.7-4.8 in the 2026-09-10 geometry audit. Sixty-four states, each 1024-dimensional,
spanning about six directions. The coda reads that channel badly and reads TOKENS well, and
a token is a DISCRETE symbol from a large alphabet with its own embedding row.

WHAT THIS BUILDS. The loop's exit state ``z`` is projected to ``K`` sub-vectors, each
sub-vector is snapped to the nearest entry of a shared codebook, and the ``K`` quantized
sub-vectors are lifted back to ``K`` full-width cells that the coda reads exactly as it
reads prefix cells today (1:1 through ``W_prefix[k]``; ``prefix_k`` must equal ``vq_codes``).
The thought a span hands forward is then ``K`` symbols out of an alphabet of ``C``, not one
point in R^d. Rank is given BY CONSTRUCTION: two spans that choose different codes are
exactly as far apart as those codes are, whatever the loop did to their continuous states.

THE FIVE DESIGN DECISIONS, each with its reason, because each is a place a reviewer would
otherwise have to guess.

1. **COSINE (l2-normalized) codes, not raw L2.** The encoder output ``W_vq(z)`` has a scale
   nobody has swept, and a codebook drawn at any fixed std either sits inside the encoder's
   cloud (fine) or far outside it (every vector snaps to one code, perplexity 1, the arm
   dies for a reason that has nothing to do with the question). Normalising BOTH sides puts
   the match on the unit sphere, where nearest-by-L2 IS nearest-by-cosine and the encoder's
   scale cannot decide the outcome. This is ViT-VQGAN's l2-normalized codebook (Yu et al.
   2022, "Vector-quantized Image Modeling with Improved VQGAN", §3.2), the standard fix for
   codebook collapse. The scale the coda needs is then supplied by ``W_vq_out``, a learned
   linear, instead of by an init nobody measured.

2. **The VQ-VAE loss (codebook + commitment), NOT an EMA codebook.** An EMA codebook is
   updated INSIDE the forward under ``no_grad``. This tree runs the forward many times per
   checkpoint with no optimiser step behind it — ``core_depth_sweep.py`` at seven forced
   depths, ``worth_profile.py`` at four ablation modes, the eval loop at 20 batches — and
   every one of those would step the codebook. The loss form keeps the codebook a plain
   parameter that moves only when ``optimizer.step()`` runs, which is the only version of
   this mechanism whose instruments mean what they say. Van den Oord et al. 2017 §3.2 gives
   both; this picks the one that is a function and not a side effect.

3. **The DEQUANTIZED thought is the MEAN of the K lifted cells.** Every reader between the
   loop and the write (the span decoder, the MUX, SIGReg, the energy) takes ONE state per
   slot. Handing them the mean keeps each of those mechanisms the SHIPPED one, so the arm
   differs from its ruler by the quantizer alone, and the span decoder grades the thought
   the coda actually gets rather than the continuous state the coda never sees. The MEAN
   and not the SUM: a sum scales with ``K``, so the K=4 and K=8 arms would differ in the
   thought's magnitude as well as in its content and the pair would not be one factor.

4. **ONE shared codebook across the K code positions (and across the Hyper-Connection
   streams).** A per-position codebook is ``K`` times the parameters for a partition the
   model could learn anyway through ``W_vq``; the shared book also lets a code that is
   useful at position 0 be reused at position 3, which is what makes the alphabet an
   alphabet. ``vq_groups > 1`` is the product-quantization axis: each cell's ``d_c`` vector
   is split into ``G`` groups of ``d_c/G`` and each group is quantized separately, so a cell
   carries ``G`` symbols and the effective per-cell alphabet is ``C^G`` at ``C`` codebook
   rows. ``G = 1`` is one symbol per cell and is the default.

5. **STEP 0 IS NOT THE RULER.** Every other TUL arm zero-inits its new path so the arm
   starts at the arm it is measured against. A quantizer cannot: a zero ``W_vq`` makes the
   encoder output zero (no direction to normalise) and a zero ``W_vq_out`` makes every
   written cell zero (the coda reads nothing at all and the model starts degenerate). So at
   step 0 this arm's coda reads ``W_vq_out[k] @ e_n``, where ``e_n`` is a near-arbitrary
   unit code picked by a random projection of ``z`` — a low-entropy signal that carries
   almost nothing about the span. The arm's early CE is therefore WORSE than its ruler's by
   construction and the comparison is at 5,000 steps, not at 200. This is stated in the
   config header and in the pre-registration; it is the price of a bottleneck.

PADS. A pad slot's ``z`` is exactly zero (``gather_valid``). Its normalised encoder output
is defined to be zero rather than ``0/0``, its recorded index is ``-1``, its cells are
zeroed, and it is excluded from both loss terms and from the usage histogram. Two-sided in
the gate: the pad's cells are zero AND every VQ parameter still receives a finite gradient
(the register's own NaN bug was found exactly here).

RNG. Every draw comes from a PRIVATE generator and the global stream is snapshotted and
restored around ``__init__`` (the ``TULSlotRegister`` precedent, where the missing snapshot
was a real leak: ``nn.Linear`` kaiming-draws on the global stream before its weight is
overwritten). So a VQ model's base weights are byte-identical to its ruler's.

PRECISION. ``W_vq`` carries ``_ternary_exclude = True`` and is not a ``MortarLinear``, and
``vq_E`` / ``W_vq_out`` are plain parameters on a non-Linear module, so the ternary QAT
scope, the CMS prune, the MORTAR carve and the deploy packer all walk past this module —
the same contract ``tul_spandec.py`` and ``tul_egrad.py`` hold.

Record: ``lab/experiments/planned/2026-09-13-arc-discrete-thought-vq.md``
Note: ``.agents/notes/proposed/architecture/2026-09-13-discrete-thought-vq.md``
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["TULThoughtVQ"]


class TULThoughtVQ(nn.Module):
    """K discrete codes per span, lifted into K prefix cells.

    Args:
        d_model:      the carrier width ``d``.
        codes:        ``K``, the number of code positions = the number of prefix cells.
        codebook:     ``C``, the number of entries in the shared codebook.
        dim:          ``d_c``, the width of a code position's sub-vector. ``0`` -> ``d // K``.
        groups:       ``G``, product-quantization groups inside one code position.
        beta:         the commitment weight (van den Oord 2017 uses 0.25).
        reset_after:  re-seed a codebook row that has gone unused for this many TRAINING
                      forwards, from the worst-served encoder vector in the current batch.
                      ``0`` = off, and off is what both shipped configs run: a dead codebook
                      is then a RESULT about the mechanism rather than a number the reset
                      manufactured. Deterministic (no draw), training-only.
    """

    def __init__(self, d_model: int, codes: int, codebook: int, dim: int,
                 groups: int, beta: float, reset_after: int = 0):
        super().__init__()
        if codes < 2:
            raise ValueError(f"TULThoughtVQ needs codes >= 2, got {codes}")
        if codebook < 2:
            raise ValueError(f"TULThoughtVQ needs codebook >= 2, got {codebook}")
        if groups < 1:
            raise ValueError(f"TULThoughtVQ needs groups >= 1, got {groups}")
        if not dim and d_model % codes:
            raise ValueError(
                f"tul.vq_codes={codes} does not divide d_model={d_model} and tul.vq_dim is "
                f"0 (which means d_model // vq_codes). Set tul.vq_dim to the width you "
                f"want rather than letting the split round.")
        d_c = int(dim) if dim else d_model // codes
        if d_c < 1:
            raise ValueError(
                f"TULThoughtVQ code width {d_c} < 1 (d_model={d_model}, codes={codes}): "
                f"pass tul.vq_dim explicitly.")
        if d_c % groups:
            raise ValueError(
                f"tul.vq_groups={groups} must divide the code width d_c={d_c} "
                f"(d_model={d_model}, vq_codes={codes}, vq_dim={dim}).")
        self.k = int(codes)
        self.c = int(codebook)
        self.d_c = int(d_c)
        self.g = int(groups)
        self.d_g = self.d_c // self.g
        self.beta = float(beta)
        self.reset_after = int(reset_after)
        self.eps = 1e-6
        # RNG-NEUTRAL CONSTRUCTION (the TULSlotRegister precedent). `nn.Linear` kaiming-draws
        # on the GLOBAL stream before the weight is overwritten, so without this snapshot
        # every parameter built after the quantizer would sit at a different point in the
        # stream than it does on the ruler.
        _rng0 = torch.random.get_rng_state()
        gen = torch.Generator(device="cpu").manual_seed(0x5E71)
        self.W_vq = nn.Linear(d_model, self.k * self.d_c, bias=False)
        with torch.no_grad():
            self.W_vq.weight.copy_(torch.empty(
                self.W_vq.weight.shape, device="cpu").normal_(
                    mean=0.0, std=0.02, generator=gen))
        # The codebook. Drawn N(0,1) and l2-normalised at LOOKUP, so the draw's scale is
        # irrelevant by construction (decision 1) and only its directions matter.
        self.vq_E = nn.Parameter(torch.empty(self.c, self.d_g).normal_(
            mean=0.0, std=1.0, generator=gen))
        # The lift: one [d_c, d] map per CODE POSITION, so position k and position 0 can
        # read the same symbol differently. std = G^-0.5 makes a lifted cell's per-component
        # rms 1.0 at init: `q` is G unit-norm groups, so var(out_j) = d_c * (G/d_c) * std^2
        # = G * std^2. 1.0 is the scale of the `input_norm`'d field the cell is scattered
        # into, so the coda's first read is neither a whisper nor a shout.
        self.W_vq_out = nn.Parameter(torch.empty(self.k, self.d_c, d_model).normal_(
            mean=0.0, std=float(self.g) ** -0.5, generator=gen))
        torch.random.set_rng_state(_rng0)
        self.W_vq._ternary_exclude = True
        # Non-persistent: it is a live counter, not a weight, and keeping it out of
        # `state_dict` keeps a VQ checkpoint's key set exactly its parameters.
        self.register_buffer("code_age", torch.zeros(self.c, dtype=torch.long),
                             persistent=False)

    def extra_repr(self) -> str:
        return (f"codes={self.k}, codebook={self.c}, d_c={self.d_c}, groups={self.g}, "
                f"beta={self.beta}, reset_after={self.reset_after}")

    def lift(self, index: Tensor) -> Tensor:
        """Symbols -> cells, no encoder: ``index`` ``[B, S, K, G]`` long (every entry in
        ``[0, C)``; a pad slot's ``-1`` is read as code 0 and the caller zeroes the slot)
        -> ``[B, S, K, C]`` fp32, the SAME map the forward applies to its quantised
        vectors (normalised codebook rows through ``W_vq_out``). This is how a SAMPLED
        symbol string reaches the coda on an LCTUL-D model (``tul.code_discrete``)."""
        if index.dim() != 4 or index.shape[2] != self.k or index.shape[3] != self.g:
            raise ValueError(
                f"TULThoughtVQ.lift wants [B, S, K={self.k}, G={self.g}] indices, got "
                f"{tuple(index.shape)}")
        e_all = self.vq_E.float()
        e_n = e_all / (e_all.norm(dim=-1, keepdim=True) + self.eps)              # [C, d_g]
        q = e_n.index_select(0, index.clamp_min(0).reshape(-1)).reshape(
            *index.shape[:3], self.d_c)                                             # [B,S,K,d_c]
        return torch.einsum("bskd,kdc->bskc", q, self.W_vq_out.float())

    def forward(self, z: Tensor, slot_valid: Tensor, sub_index: Tensor | None = None
                ) -> tuple[Tensor, Tensor, dict]:
        """``z`` ``[B, S, *mid, C]`` -> ``(cells [B, S, K, *mid, C], dequant, out)``.

        ``sub_index`` ``[B, S, *mid, K, G]`` long, optional: where it is ``>= 0`` the
        quantised vector is REPLACED by that codebook row before the lift (LaDiR's token
        substitution, LCTUL-D's rate control) and the straight-through gradient is cut at
        that symbol (the encoder did not produce it, so it must not be told it did).
        ``out["index"]`` still reports the encoder's own assignment.

        ``mid`` is ``()`` on a plain carrier and ``(n,)`` on the Hyper-Connection carrier;
        the quantizer is a per-stream map exactly as every other ``nn.Linear`` in the model
        is, so an ``n``-stream model carries ``n*K*G`` symbols per span and the coda's cell
        ``k`` holds stream ``j``'s code ``k`` in stream ``j``.

        ``out`` carries ``loss`` (the VQ term, still attached), ``commit`` / ``codebook``
        (the two halves, attached), ``vq_perplexity`` / ``vq_used`` (floats over the batch's
        VALID assignments) and ``index`` ``[B, S, *mid, K, G]`` with ``-1`` at every pad.
        """
        B, S = slot_valid.shape
        mid = tuple(z.shape[2:-1])
        # [B, S, *mid, K, G, d_g] in FP32. The `W_vq` matmul itself runs in the autocast
        # dtype, as every other Linear here does, but everything downstream of it is pinned
        # to fp32 on purpose: the normalisation divides by a norm, the assignment is an
        # `argmax` over cosine similarities that a bf16 rounding can flip between two close
        # codes, and the two loss terms reduce ~12.6 M entries at the arm's shape. A
        # discrete choice made in bf16 would make a run's code assignments depend on the
        # rounding rather than on the state. On an fp32 forward `.float()` is a no-op, so
        # the CPU gate's numbers are unchanged.
        u = self.W_vq(z).float().reshape(B, S, *mid, self.k, self.g, self.d_g)
        # Unit sphere. `+ eps` (not a clamp on the norm) keeps a pad's exact zero an exact
        # zero instead of a unit vector pointing wherever the numerics landed.
        u_n = u / (u.norm(dim=-1, keepdim=True) + self.eps)
        e_all = self.vq_E.float()
        e_n = e_all / (e_all.norm(dim=-1, keepdim=True) + self.eps)        # [C, d_g]
        # Nearest by L2 on the sphere == largest cosine. One matmul, no [N, C, d_g] tensor.
        sim = torch.matmul(u_n.reshape(-1, self.d_g), e_n.t())             # [N, C]
        idx = sim.argmax(dim=-1)                                           # [N]
        q_e = e_n.index_select(0, idx).reshape_as(u_n)
        # [B, S, 1...1] -> broadcast over mid, K, G
        vmask = slot_valid.reshape(B, S, *([1] * (u_n.dim() - 3)))
        vf = vmask.to(u_n.dtype)
        denom = (vf.expand(u_n.shape[:-1]).sum() * self.d_g).clamp(min=1.0)
        # Both terms on the NORMALISED vectors: the commitment must not become a constraint
        # on ||W_vq(z)||, which nobody asked for and which the lift can undo anyway.
        codebook = (((u_n.detach() - q_e) ** 2).sum(dim=-1) * vf).sum() / denom
        commit = (((u_n - q_e.detach()) ** 2).sum(dim=-1) * vf).sum() / denom
        vq_loss = codebook + self.beta * commit
        # The straight-through estimator: the VALUE is the code, the GRADIENT is the
        # encoder's. It is the only edge by which the TOKEN CE and the SPAN DECODER reach
        # the loop on a VQ model. It is NOT the only edge into the loop, and the difference
        # is measured: the commitment term below also reaches `u_n` -> `W_vq` -> `z`, so at
        # `vq_weight > 0` a detached estimator still leaves the core a nonzero gradient
        # (sabotage S2 MISSED a test that assumed otherwise). The two-sided gate for this
        # line runs at `vq_weight: 0`, where the STE IS the only route.
        q = u_n + (q_e - u_n).detach()
        if sub_index is not None:
            if sub_index.shape != u_n.shape[:-1]:
                raise ValueError(
                    f"sub_index {tuple(sub_index.shape)} must match the symbol grid "
                    f"{tuple(u_n.shape[:-1])}")
            sub = sub_index >= 0
            q_sub = e_n.index_select(0, sub_index.clamp_min(0).reshape(-1)).reshape_as(u_n)
            q = torch.where(sub.unsqueeze(-1), q_sub.detach(), q)
        q = q.reshape(B, S, *mid, self.k, self.d_c).to(z.dtype)
        cells = torch.einsum("...kd,kdc->...kc", q, self.W_vq_out.to(q.dtype))
        cells = cells.movedim(-2, 2)                                   # [B, S, K, *mid, C]
        cells = cells * slot_valid.reshape(
            B, S, *([1] * (cells.dim() - 2))).to(cells.dtype)
        # Decision 3: the readers between here and the write take ONE state per slot.
        dequant = cells.mean(dim=2)                                    # [B, S, *mid, C]
        vflat = vmask.expand(u_n.shape[:-1]).reshape(-1)
        counts = torch.bincount(idx[vflat], minlength=self.c).to(u_n.dtype)
        p = counts / counts.sum().clamp(min=1.0)
        ppl = torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum())
        out = {
            # fp32 by construction (see the `.float()` above); the caller's weighted twin
            # and the total loss promote to it, which is what every other auxiliary term
            # in this forward already does.
            "loss": vq_loss,
            "commit": commit,
            "codebook": codebook,
            "vq_perplexity": float(ppl),
            "vq_used": float((counts > 0).sum()),
            "vq_n_codes": float(self.k * self.g),
            "vq_codebook_size": float(self.c),
        }
        if self.reset_after > 0 and self.training:
            self._reset_dead(counts, u_n.detach(), sim.detach(), idx, vflat)
        shaped = idx.reshape(u_n.shape[:-1])
        out["index"] = torch.where(vmask.expand_as(shaped), shaped,
                                   torch.full_like(shaped, -1))
        return cells, dequant, out

    @torch.no_grad()
    def _reset_dead(self, counts: Tensor, u_n: Tensor, sim: Tensor, idx: Tensor,
                    vflat: Tensor) -> None:
        """Re-seed codebook rows unused for ``reset_after`` training forwards.

        DETERMINISTIC on purpose: the replacements are the VALID encoder vectors this batch
        served WORST (lowest cosine to their own assigned code), so the reset draws no
        random number at all and cannot move the model's RNG stream. A random pick would
        have to run on the model's device, where a private CPU generator cannot reach.
        """
        self.code_age = torch.where(counts > 0, torch.zeros_like(self.code_age),
                                    self.code_age + 1)
        dead = (self.code_age >= self.reset_after).nonzero(as_tuple=True)[0]
        n_valid = int(vflat.sum())
        n = int(min(dead.numel(), n_valid))
        if n == 0:
            return
        err = 1.0 - sim.gather(1, idx.unsqueeze(1)).squeeze(1)              # [N]
        err = torch.where(vflat, err, torch.full_like(err, -1.0))
        pick = err.topk(n).indices
        src = u_n.reshape(-1, self.d_g).index_select(0, pick)
        self.vq_E.data.index_copy_(0, dead[:n], src.to(self.vq_E.dtype))
        self.code_age.index_fill_(0, dead[:n], 0)
