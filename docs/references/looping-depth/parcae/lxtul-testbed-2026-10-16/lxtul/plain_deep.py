"""HRM-style deep supervision on the plain Parcae loop (`plain.deep_supervision`, default off; the plain path is then
untouched). The plain arm draws a per-row pass count (Parcae's law, `_sample_batch_depths`), truncated at
`plain.deep_max_passes`. Here EVERY pass is followed by the coda and the head, which take the main CE; the core state
carried into the next pass is DETACHED, so each pass's CE trains that pass (and the prelude) only. A row's NLL is the
mean over its own passes, so the loss is the mean of the per-pass CEs. Training only: eval runs the ordinary forward.
This isolates deep supervision from the spiral's moving hierarchy (`spiral.cycle.credit: deep` is the same recipe on the
cycle). The caller (`PlainPointer`) supplies the head and does the weighting.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from lxtul.ce import linear_ce_tokens


def head_logits(model: nn.Module, x: Tensor) -> Tensor:
    """fp32 logits of the tied head on the final-norm output x, with Parcae's scale and softcap (the return_logits
    branch of `Parcae.forward`)."""
    cfg = model.config
    if cfg.use_fused_head == "full-triton":
        logits = torch.matmul(x, model.lm_head.weight.T if cfg.tie_embeddings else model.lm_head.weight
                              ).float() * cfg.init.logit_scale
    else:
        logits = model.lm_head(x).float() * cfg.init.logit_scale
    if cfg.logit_softcap is not None:
        logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)
    return logits


def head_nll(model: nn.Module, h: Tensor, labels: Tensor) -> Tensor:
    """Per-position CE [B, L] (0 where label is -100) of the tied head on h, by the chunked linear CE of lxtul/ce.py: no
    [B, L, V] logits tensor is kept (the deep heads run once per pass or cycle). Same function as CE of `head_logits`
    (no masked logit); the GEMMs run in bf16 like the head's own, fp32 accumulation."""
    cfg = model.config
    if cfg.logit_softcap is not None:
        raise ValueError("the chunked deep head has no logit softcap; use a model without one")
    ce = linear_ce_tokens(h.reshape(-1, h.shape[-1]), model.lm_head.weight, labels.reshape(-1),
                          float(cfg.init.logit_scale), -1)
    return ce.view(labels.shape)


def carry(x: Tensor) -> Tensor:
    """The core state carried into the next pass: DETACHED, so a pass's CE trains that pass only."""
    return x.detach()


def check_plain_deep(model: nn.Module) -> None:
    cfg = model.config
    if cfg.recurrent_iteration_method != "per-sequence":
        raise ValueError(f"plain.deep_supervision needs the per-sequence sampler, got {cfg.recurrent_iteration_method!r}")
    if model.gradient_checkpointing or cfg.row_compact or getattr(model, "sync_free_loop", False):
        raise ValueError("plain.deep_supervision runs its own loop: gradient_checkpointing, row_compact and sync_free "
                         "must be off")
    if cfg.randomize_positions_from is not None:
        raise ValueError("plain.deep_supervision does not support randomize_positions_from")


def deep_forward(model: nn.Module, input_ids: Tensor, head_fn, max_passes: int, backward_scale: Tensor | None = None) -> dict:
    """With `backward_scale` (training with grad on) each pass's weighted CE, sum(weight * nll) / backward_scale, is
    BACKPROPAGATED before the next pass, so the graph of a pass is freed at once (memory of one pass, as HRM does it);
    the prelude output is a leaf during the loop and its accumulated gradient flows into the prelude after the last
    pass. The summed gradient equals that of one backward through the sum of the pass losses. The returned NLLs are then
    detached. -> {"cycle_nll": [per-pass NLL [B, L]], "cycle_weight": [P, B] (1 / the row's pass count while it runs, else 0),
    "cycle_active": [P, B] bool}. `head_fn(h)` maps the final-norm output to the NLL [B, L] with grad."""
    B, L = input_ids.shape
    freqs = model.freqs_cis[:, :L]
    tr = model.transformer
    e = tr.wte(input_ids)
    if model.emb_scale != 1:
        e = e * model.emb_scale
    model._current_input_ids = input_ids
    for i, block in enumerate(tr.prelude):
        ve = model.value_embeds[str(i)](input_ids) if str(i) in model.value_embeds else None
        e = block(e, freqs, None, ve=ve)
    if model.config.prelude_norm:
        e = tr.ln_prelude(e)
    n, k = model._sample_batch_depths(B, None)               # CPU
    total = (n + k).clamp(1, max_passes)
    total_dev = total.to(e.device, non_blocking=True)
    e_in = e
    if backward_scale is not None:
        e = e.detach().requires_grad_()
    coda_off = model.config.n_layers_in_prelude + model.config.n_layers_in_recurrent_block
    x = model.initialize_state(e)
    nlls, active = [], []
    for c in range(int(total.max())):
        on = c < total_dev
        x_new = model.core_block_forward(x, e, freqs, None, torch.tensor(c, device=e.device), total_dev)
        x = torch.where(on[:, None, None], x_new, x)
        y = tr.C(x)
        for i, block in enumerate(tr.coda):
            ve = model.value_embeds[str(coda_off + i)](input_ids) if str(coda_off + i) in model.value_embeds else None
            y = block(y, freqs, None, ve=ve)
        nll = head_fn(tr.ln_f(y))
        active.append(on)
        if backward_scale is not None:
            ((on.float() / total_dev.float())[:, None] * nll).sum().div(backward_scale).backward()
            nll = nll.detach()
        nlls.append(nll)
        x = carry(x)
    if backward_scale is not None:
        torch.autograd.backward([e_in], [e.grad])
    active = torch.stack(active)
    return {"cycle_nll": nlls, "cycle_active": active,
            "cycle_weight": active.float() / total_dev.float()[None]}
