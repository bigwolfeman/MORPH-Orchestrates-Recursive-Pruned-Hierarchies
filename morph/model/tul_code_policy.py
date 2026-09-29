"""Arm B, the code policy (``tul.code_policy_k = C > 1``, 2026-09-29): explore ACROSS
training steps with ONE rollout instead of across K parallel rollouts.

LX (``tul.code_enum_k``, :mod:`morph.model.tul_code_enum`) runs K rollouts of the slot loop
and the coda per row, rollout k re-adding a learned code ``u_k`` at the end of every pass,
and trains the exact per-span mixture. It costs about K codas, and its gain was measured to
be mostly an ensemble: a fixed code does not specialise per span
(``lab/experiments/failures/2026-09-26-lx-selection-ceiling.md``).

This arm keeps the codes and runs ONE rollout:

* THE CODES are LX's own module, :class:`~morph.model.tul_code_enum.TULCodeEnum` with ``C``
  vertices (a learned regular simplex: unit RMS, sum zero, equidistant, no learned scale),
  re-added by the SAME rule ``h <- f(h) + r * rms(f(h)).detach() * u_c`` at the SAME place
  in ``MORPHTransformer._tul_core`` (search "tul.code_policy_k"), indexed per SLOT
  (:meth:`TULCodeEnum.term_per_slot`) instead of per rollout. Pads get nothing.
* THE POLICY is a linear head ``d_model -> C`` on the slot's loop-ENTRY state: ``e``, the
  prelude's output gathered at the slot position after ``input_norm`` (the tensor
  ``core_init`` turns into the first carrier), stream-averaged, DETACHED and scaled to unit
  L2 norm (:meth:`TULCodePolicy.features`). THE VALUE head reads the same input.
* THE CHOICE: at train one code per valid slot is sampled from ``softmax(logits)`` with the
  GLOBAL RNG stream (``torch.rand`` on the model's device: an arm with the key on does not
  share its later random draws with its control). At eval ``tul.code_policy_eval`` decides:
  ``argmax`` (default) or ``sample``; an eval ``sample`` and the val-only ``random`` pick
  draw from a PRIVATE generator, so an eval pass moves no global stream.
* THE REWARD of slot ``s`` is ``r_s = -`` the MEAN token CE (nats per scored token) of the
  span slot s's cell feeds: bag ``s + 1`` of :func:`morph.model.transformer.span_ce_index`
  (the positions whose predictions can first read slot s's cell under the strict geometry:
  span s+1's tokens, the last of which predicts span s+2's first token). The CE is the
  model's own per-token CE (the slot id masked from the partition, as the trained loss
  masks it), computed under ``no_grad`` from the coda state. Slots whose span has no scored
  token are masked out.
* THE BASELINE is ``b_s = rbar_{-s} + v_s``: the leave-one-out mean reward of the batch's
  other valid slots plus the value head's prediction of the residual. WHY not ``b_s = v_s``
  alone: the reward sits near ``-log`` perplexity (about -4 to -7 nats), a zero-init head
  starts 4-7 nats off, and (a) its squared-error gradient on the bias (about 10 at step 0)
  would dominate the trainer's GLOBAL gradient clip (``training.grad_clip`` 1.0) and shrink
  every other parameter's update, and (b) at lr 1e-4 a normalised optimizer moves the bias
  about 1e-4 per step, so the head would need tens of thousands of steps to reach the
  reward's level. The residual has mean about 0 and a spread of about 1 nat, which the
  head can learn and which keeps its gradient small. The leave-one-out mean keeps slot
  s's own reward out of its baseline.
* THE LOSSES (averaged over the N scored slots; ``A_s = r_s - b_s`` detached):
  ``code_policy_lambda * -A_s log pi(c_s)``  ``- code_policy_entropy * H(pi_s)``
  ``+ code_policy_value_lambda * (v_s - (r_s - rbar_{-s}))^2``, folded as ONE
  ``code_policy_weighted`` term at TRAIN only (train.py subtracts it from train/loss and
  from the val loss). The REINFORCE gradient reaches the policy and value heads ONLY (their
  input is detached). The loop and the codes ``u_c`` still learn from the ordinary token CE
  through the one coda pass: the codes are trained by the CE, the choice by REINFORCE.

FOOT GUNS (read before trusting a reading):

(a) REWARD COUPLING. Under the strict geometry a LATER span's tokens read EARLIER slots'
    cells, so slot s's code also moves the CE of spans s+2, s+3, ... which its per-span
    reward does not credit (and the leave-one-out baseline therefore has an O(1/N)
    dependence on c_s through those spans). The estimator is the per-span credit, not the
    full return.
(b) POLICY COLLAPSE to one code: watch ``code_policy_share_k{i}`` and
    ``code_policy_entropy`` (normalised by log C: 1.0 uniform, 0 deterministic). The
    entropy bonus is the only force against it.
(c) TRAIN SAMPLES, EVAL TAKES THE ARGMAX: a train/deploy gap by construction.
    ``tul.code_policy_eval: sample`` exists to measure it; the trainer logs
    ``val/code_policy_sample_minus_argmax`` under that mode.
(d) THE POLICY CHOOSES FROM THE PRELUDE'S SUMMARY ONLY: it reads the slot before any pass
    of the loop, so it cannot see what the loop would make of the slot.
(e) REINFORCE VARIANCE: hundreds of slots per batch, ONE sample each. The decisive
    instrument is ``val/code_policy_vs_random`` (train.py's val-only extra pass): token
    CE with a uniformly random code per slot minus token CE with the policy's argmax, in
    nats per token. About 0 means the policy is picking noise.

Never ternarised (``_ternary_exclude`` on every leaf), invisible to prune/carve/route (no
``MortarLinear``), RNG-neutral construction: the codes' basis comes from ``TULCodeEnum``'s
private generator and the two heads are built inside a forked stream and zeroed, so every
other weight equals the same-seed model without the key.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .tul_code_enum import TULCodeEnum

_BUILD_SEED = 0xB0C0DE
_EVAL_SEED = 0xE7A1C0DE


class TULCodePolicy(nn.Module):
    """C codes (LX's simplex), a policy head and a value head (module doc)."""

    def __init__(self, d_model: int, k: int, ratio: float):
        super().__init__()
        if k < 2:
            raise ValueError(f"TULCodePolicy needs k >= 2 codes, got {k}")
        self.k = int(k)
        self.codes = TULCodeEnum(int(d_model), self.k, float(ratio))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_BUILD_SEED)
            self.policy = nn.Linear(int(d_model), self.k)
            self.value = nn.Linear(int(d_model), 1)
        with torch.no_grad():
            # Zero init: the policy starts UNIFORM (entropy log C, no code preferred) and
            # the value head starts at the batch-mean baseline. Neither is a symmetric
            # saddle: the REINFORCE gradient of a zero head is A (onehot - 1/C) x^T.
            for lin in (self.policy, self.value):
                lin.weight.zero_()
                lin.bias.zero_()
        self._ternary_exclude = True
        for m in (self.codes, self.policy, self.value):
            m._ternary_exclude = True
        # Val-only draws (eval `sample`, the `random` pick) count up from _EVAL_SEED so two
        # val batches get different draws and a rerun gets the same ones. Not a buffer: it
        # is bookkeeping for an instrument, never part of the model.
        self._eval_draws = 0

    @staticmethod
    def features(e: Tensor) -> Tensor:
        """``[B, S, C]`` fp32 policy input: the loop-entry state ``e`` ``[B, S, (n,) C]``
        DETACHED, averaged over the Hyper-Connection streams, scaled to unit L2 norm. The
        norm keeps the heads' raw gradients O(1) whatever the carrier's scale (the global
        clip sees them); a pad slot (``e`` exactly 0) stays 0."""
        x = e.detach().float()
        if x.dim() == 4:
            x = x.mean(dim=2)
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def heads(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(logits [B, S, C], value [B, S])`` in fp32 with autocast off."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x, self.policy.weight.float(), self.policy.bias.float())
            value = F.linear(x, self.value.weight.float(), self.value.bias.float())
        return logits, value.squeeze(-1)

    @staticmethod
    def sample(logits: Tensor, u: Tensor) -> Tensor:
        """``[B, S]`` int64 codes drawn from ``softmax(logits)`` by the inverse CDF of the
        uniforms ``u`` ``[B, S]`` in [0, 1). ONE home for every categorical draw here (the
        train draw, the eval ``sample``), so a test can force the choice by patching it."""
        cdf = torch.softmax(logits.float(), dim=-1).cumsum(dim=-1)
        c = (cdf <= u.unsqueeze(-1).to(cdf.dtype)).sum(dim=-1)
        return c.clamp(max=logits.shape[-1] - 1)

    def eval_uniform(self, shape, device) -> Tensor:
        """``[B, S]`` uniforms from a PRIVATE generator (CPU, then moved): an eval draw
        never moves the global RNG stream."""
        g = torch.Generator(device="cpu").manual_seed(_EVAL_SEED + self._eval_draws)
        self._eval_draws += 1
        return torch.rand(tuple(shape), generator=g).to(device)

    def random_codes(self, shape, device) -> Tensor:
        """``[B, S]`` int64 codes uniform over the C codes, from the private stream: the
        val-only ``random`` pick behind ``val/code_policy_vs_random``."""
        u = self.eval_uniform(shape, device)
        return (u * self.k).long().clamp(max=self.k - 1)


def span_mean_ce(tok_ce: Tensor, gid: Tensor, keep_tok: Tensor,
                 n_groups: int) -> tuple[Tensor, Tensor]:
    """``(mean_ce [B, S], has [B, S])``: the MEAN token CE per bag, read for slot s at bag
    s+1 (the span its cell feeds), and whether that bag has any scored token.

    ``tok_ce`` ``[B, L]`` per-position CE, ``gid`` / ``keep_tok`` / ``n_groups`` from
    :func:`morph.model.transformer.span_ce_index` (bins ``b * G + bag``, unscored positions
    dumped into bin 0 with an exactly-zero weight here). A bag with no scored token reads
    0 and ``has`` False."""
    B = tok_ce.shape[0]
    w = keep_tok.to(torch.float32)
    sums = torch.zeros(B * n_groups, device=tok_ce.device, dtype=torch.float32)
    cnts = torch.zeros_like(sums)
    sums.index_add_(0, gid.reshape(-1), (tok_ce.float() * w).reshape(-1))
    cnts.index_add_(0, gid.reshape(-1), w.reshape(-1))
    sums, cnts = sums.view(B, n_groups)[:, 1:], cnts.view(B, n_groups)[:, 1:]
    return sums / cnts.clamp(min=1.0), cnts > 0


def policy_objective(logits: Tensor, value: Tensor, codes: Tensor, reward: Tensor,
                     mask: Tensor, lam: float, ent_w: float,
                     value_w: float) -> tuple[Tensor, dict]:
    """The weighted REINFORCE + entropy + baseline objective and its readings.

    ``logits`` ``[B, S, C]`` and ``value`` ``[B, S]`` carry gradient (to the heads only),
    ``codes`` ``[B, S]`` the chosen codes, ``reward`` ``[B, S]`` (detached here), ``mask``
    ``[B, S]`` the slots that count (valid AND their span has a scored token). Means over
    the N counted slots; at N = 0 the objective is an exact 0 that keeps the graph.

    Returns ``(weighted_loss, stats)``; every stat is a detached 0-dim fp32 tensor (no host
    sync on the training path)."""
    C = logits.shape[-1]
    m = mask.to(torch.float32)
    n = m.sum()
    denom = n.clamp(min=1.0)
    r = reward.detach().float() * m
    # Leave-one-out batch mean: slot s's own reward never enters its baseline.
    rbar = torch.where(n > 1.0, (r.sum() - r) / (n - 1.0).clamp(min=1.0),
                       torch.zeros_like(r))
    target = (r - rbar) * m                                  # the value head's target
    v = value.float()
    adv = ((r - rbar - v).detach()) * m                      # A_s = r_s - b_s
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    logp_c = logp.gather(-1, codes.long().unsqueeze(-1)).squeeze(-1)
    ent = -(p * logp).sum(dim=-1)                            # [B, S] nats
    pg = -(adv * logp_c * m).sum() / denom
    ent_mean = (ent * m).sum() / denom
    vmse = (((v - target) ** 2) * m).sum() / denom
    loss = lam * pg - ent_w * ent_mean + value_w * vmse
    with torch.no_grad():
        chosen = F.one_hot(codes.long(), C).float() * m.unsqueeze(-1)
        share = chosen.sum(dim=(0, 1)) / denom
        agree = ((codes == logits.argmax(dim=-1)).float() * m).sum() / denom
        adv_mean = adv.sum() / denom
        adv_std = (((adv - adv_mean) ** 2) * m).sum().div(denom).sqrt()
        stats = {
            "code_policy_pg": pg.detach(),
            "code_policy_entropy": (ent_mean / math.log(C)).detach(),
            "code_policy_adv_mean": adv_mean,
            "code_policy_adv_std": adv_std,
            "code_policy_value_mse": vmse.detach(),
            "code_policy_reward_mean": r.sum() / denom,
            "code_policy_agree": agree,
            "code_policy_n_slots": n.detach(),
        }
        for i in range(C):
            stats[f"code_policy_share_k{i}"] = share[i]
    return loss, stats
