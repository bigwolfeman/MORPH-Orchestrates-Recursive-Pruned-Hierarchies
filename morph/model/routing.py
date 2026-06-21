"""ReMoE-style hidden-neuron routing (Phase C).

Design:
  TileRouter — per-token product-key routing over neuron-clusters of the d_ff
               hidden bank.  Uses PEER-style product keys: two sub-codebooks +
               Cartesian product.  Differentiable via continuous soft gates
               (ReLU-based, no STE needed).  Built-in load-balancing aux_loss
               to prevent routing collapse.

The router is hosted by _SwiGLUMortar (transformer.py): it gates the post-SiLU
hidden h = silu(gate)·up over contiguous d_ff neuron-clusters, BEFORE the down
projection — fully backend-agnostic (works on dense and MORTAR-carved MLPs
alike; the carved BCSR GEMM is untouched).  Routing is activated by
PruningSchedule._activate_routing at route_start.

"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# TileRouter — product-key routing over tile-groups
# =============================================================================


class TileRouter(nn.Module):
    """Per-token routing over tile-groups via PEER-style product keys.

    A tile-group is a hardware execution block (cluster of row-groups).
    The router outputs a continuous gate per tile-group per token.
    Inactive groups get gate=0; active groups get a positive soft weight.

    Architecture (PEER-adapted):
    1. Project input to query: [B, T, d_model] → [B, T, d_model]
    2. Split query into two halves: q_a, q_b  ([B, T, d_model//2] each)
    3. Score against two sub-codebooks (n_sub_keys × d_model//2 each)
       - scores_a = q_a @ sub_keys_a.T  → [B, T, n_sub_keys]
       - scores_b = q_b @ sub_keys_b.T  → [B, T, n_sub_keys]
    4. Cartesian product top-k: find top-(activation_k) pairs (i, j) by scores_a[i]*scores_b[j]
       - Efficient: take top-sqrt(k) in each sub-space, combine n_sub_keys² candidates.
    5. Gather tile-group keys for selected pairs → logits
    6. ReLU gate: gate = relu(logit) → continuous, differentiable, naturally sparse
    7. Normalize active gates to sum to n_active (preserves output magnitude)

    Load balancing:
    The aux_loss minimises the variance of per-group activation counts over a
    rolling buffer.  No auxiliary coefficient hyperparameter — it auto-scales.

    Args:
        n_tile_groups:   Total number of tile-groups (= n_clusters from CMSBlockLinear).
        d_model:         Input feature dimension.
        activation_ratio: Fraction of tile-groups activated per token (target).
                         Actual activation is soft — this sets the expected value.
        n_sub_keys:      Sub-codebook size.  Should be ~ceil(sqrt(n_tile_groups)).
        aux_loss_coeff:  Coefficient for load-balance auxiliary loss.

    Outputs:
        forward returns (gates, aux_loss) where:
          - gates:    [B, T, n_tile_groups] float, non-negative.  Active groups
                      have positive values, inactive groups are exactly 0.
          - aux_loss: scalar, added to training loss for load balancing.
    """

    def __init__(
        self,
        n_tile_groups: int,
        d_model: int,
        activation_ratio: float = 0.5,
        n_sub_keys: int = 0,
        aux_loss_coeff: float = 1e-2,
        n_iters: int = 1,
    ) -> None:
        super().__init__()

        self.n_tile_groups = n_tile_groups
        self.d_model = d_model
        self.activation_ratio = activation_ratio
        self.activation_k = max(1, round(activation_ratio * n_tile_groups))
        self.aux_loss_coeff = aux_loss_coeff

        # Auto-set n_sub_keys if not specified
        if n_sub_keys <= 0:
            n_sub_keys = max(4, math.ceil(math.sqrt(n_tile_groups)))
        self.n_sub_keys = n_sub_keys

        d_half = d_model // 2

        # Query projection: d_model → d_model
        self.query_proj = nn.Linear(d_model, d_model, bias=False)

        # Product-key sub-codebooks: [n_sub_keys, d_half]
        # Initialized small so initial routing is near-uniform (all groups ~equal score)
        self.sub_keys_a = nn.Parameter(
            torch.randn(n_sub_keys, d_half) * (d_half ** -0.5)
        )
        self.sub_keys_b = nn.Parameter(
            torch.randn(n_sub_keys, d_half) * (d_half ** -0.5)
        )

        # Tile-group key vectors: [n_tile_groups, 1]
        # Maps (sub_a, sub_b) index pair → tile-group logit
        # We store a flat [n_sub_keys * n_sub_keys] → n_tile_groups projection
        # when n_sub_keys² != n_tile_groups; or a direct 1:1 key when they match.
        # For simplicity: per-tile-group learnable bias (logit offset)
        self.group_bias = nn.Parameter(torch.zeros(n_tile_groups))

        # Iteration-aware conditioning (spec Phase C: "different loop iterations activate
        # different tile groups"). A learnable embedding per CORE-LOOP iteration is added to
        # the projected query before the norm, so one shared router specializes its routing by
        # which loop iteration t it is called in. ZERO-INIT → bit-identical to a non-iteration-
        # aware router at init; specialization emerges through training. iter_idx clamps to range.
        self.n_iters = max(1, int(n_iters))
        self.iter_embed = nn.Parameter(torch.zeros(self.n_iters, d_model))

        # Query LayerNorm (PEER paper showed this critical for utilization)
        self.query_norm = nn.LayerNorm(d_model)

        # Load balance tracking: rolling mean of per-group activation rates
        # Not a gradient target — we track this via EMA to compute the aux_loss
        self.register_buffer(
            "group_load_ema",
            torch.ones(n_tile_groups) / n_tile_groups,
        )
        self._load_ema_alpha = 0.99  # slow EMA for stable reference

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: Tensor, iter_idx: int = 0) -> Tuple[Tensor, Tensor]:
        """Compute per-token routing gates.

        Args:
            x: [B, T, d_model] or [B*T, d_model] input activations.
            iter_idx: which core-loop iteration this call belongs to (selects the iteration
                      embedding). Clamped to [0, n_iters-1]. Default 0 (+ zero-init embed) →
                      identical to a non-iteration-aware router.

        Returns:
            gates:    [B, T, n_tile_groups] — non-negative continuous weights.
                      Zero for inactive groups, positive for active groups.
                      Active groups sum to n_tile_groups * activation_ratio per token.
            aux_loss: scalar — load-balance penalty (add to training loss).
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x
            B, T = x_flat.shape[0], 1

        # 1. Project query, add the loop-iteration embedding, then normalize.
        # Cast to router dtype (router parameters default to fp32 unless moved to bf16)
        proj_dtype = self.query_proj.weight.dtype
        _it = max(0, min(int(iter_idx), self.n_iters - 1))  # clamp into the embedding table
        q_proj = self.query_proj(x_flat.to(proj_dtype)) + self.iter_embed[_it].to(proj_dtype)
        q = self.query_norm(q_proj)  # [N, d_model]
        d_half = self.d_model // 2
        q_a = q[:, :d_half]   # [N, d_half]
        q_b = q[:, d_half:]   # [N, d_half]

        # 2. Sub-key scores
        # [N, n_sub_keys]
        scores_a = q_a @ self.sub_keys_a.T  # [N, n_sub_keys]
        scores_b = q_b @ self.sub_keys_b.T  # [N, n_sub_keys]

        # 3. Product-key full scores: Cartesian product
        # [N, n_sub_keys, 1] + [N, 1, n_sub_keys] → [N, n_sub_keys²]
        product_scores = (
            scores_a.unsqueeze(2) + scores_b.unsqueeze(1)
        ).reshape(x_flat.shape[0], self.n_sub_keys * self.n_sub_keys)  # [N, n_sub_keys²]

        # 4. Map product indices to tile-group logits
        # We need a [n_sub_keys², n_tile_groups] projection.
        # Implementation: learn a linear map from product space to group logits.
        # To keep this lightweight we use the group_bias as a direct logit over
        # n_tile_groups, and project down if n_sub_keys² != n_tile_groups.
        n_products = self.n_sub_keys * self.n_sub_keys

        if n_products == self.n_tile_groups:
            # Direct 1:1 mapping
            group_logits = product_scores + self.group_bias.unsqueeze(0)
        elif n_products >= self.n_tile_groups:
            # Take top-n_tile_groups product scores (fast: linear in n_tile_groups)
            # Use topk to select the strongest n_tile_groups product entries
            top_scores, top_idx = product_scores.topk(self.n_tile_groups, dim=-1)  # [N, G]
            # Map to group logits via a simple sum reduction over selected products
            group_logits = top_scores + self.group_bias.unsqueeze(0)
        else:
            # More groups than products: broadcast product scores across groups
            # Expand product scores to group space via stride-based index wrap
            idx = torch.arange(self.n_tile_groups, device=x.device) % n_products
            group_logits = product_scores[:, idx] + self.group_bias.unsqueeze(0)

        # 5. Continuous soft gate via ReLU (ReMoE-style)
        # Top-k routing: zero out all but top-activation_k groups, then ReLU
        # This preserves gradients to the top-k groups while zeroing the rest.
        if self.activation_k < self.n_tile_groups:
            # Find the k-th largest value as the threshold
            kth_vals, _ = group_logits.topk(self.activation_k, dim=-1)  # [N, k]
            threshold = kth_vals[:, -1].unsqueeze(-1)  # [N, 1]
            masked_logits = group_logits - threshold    # shift: top-k ≥ 0, rest < 0
        else:
            masked_logits = group_logits

        gates = F.relu(masked_logits)  # [N, n_tile_groups] — sparse, continuous

        # 6. Normalize active gates so they sum to activation_k per token
        # This keeps the output magnitude stable independent of how many groups fire
        gate_sum = gates.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        gates = gates * (self.activation_k / gate_sum)

        # 7. Reshape back
        if len(orig_shape) == 3:
            gates = gates.reshape(B, T, self.n_tile_groups)

        # 8. Aux loss: load balance
        # Compute mean activation rate per group in this batch
        with torch.no_grad():
            # Per-group mean activation (fraction of tokens that activate it)
            batch_load = (gates > 0).float().reshape(-1, self.n_tile_groups).mean(0)
            # Update EMA (no gradient)
            self.group_load_ema.mul_(self._load_ema_alpha).add_(
                batch_load, alpha=1 - self._load_ema_alpha
            )

        # Aux loss: variance of per-group gate sums within this batch
        # Minimizing variance encourages uniform load without a fixed target
        batch_gate_mean = gates.reshape(-1, self.n_tile_groups).mean(0)  # [G]
        aux_loss = self.aux_loss_coeff * batch_gate_mean.var()

        return gates, aux_loss

    def log_stats(self) -> Dict[str, float]:
        """Return dict of routing diagnostics for wandb.

        Returns:
            dict with keys: utilization (fraction of groups used > threshold),
            load_std (load imbalance), entropy (routing entropy estimate).
        """
        with torch.no_grad():
            load = self.group_load_ema
            used = (load > 1e-4).float().mean().item()
            load_std = load.std().item()
            # Entropy over EMA load (soft estimate)
            p = load / load.sum().clamp(min=1e-8)
            entropy = -(p * (p + 1e-8).log()).sum().item()
            max_entropy = math.log(self.n_tile_groups)

        return {
            "router/utilization": used,
            "router/load_std": load_std,
            "router/entropy_normalized": entropy / max(max_entropy, 1e-8),
        }


# =============================================================================
# Utilities for the training loop
# =============================================================================


def collect_routing_aux_losses(model: nn.Module) -> Tensor:
    """Collect and clear stashed aux_losses from all routed modules.

    Call this AFTER model.forward() and BEFORE loss.backward().
    Returns a scalar tensor (sum of all aux losses), or 0 if no routed layers.

    Generic scan: any module that stashes a non-None `_last_aux_loss` is collected and
    cleared. This covers the _SwiGLUMortar hidden-neuron router used by the Phase-C
    looped core — routing.py stays decoupled from transformer.py (no import) by keying
    on the attribute, not the type.
    """
    total: Tensor | None = None
    for m in model.modules():
        aux = getattr(m, "_last_aux_loss", None)
        if aux is not None:
            total = aux if total is None else total + aux
            m._last_aux_loss = None
    if total is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return total


def collect_routing_stats(model: nn.Module) -> Dict[str, float]:
    """Collect routing diagnostics from all TileRouters for wandb logging.

    Scans for TileRouter instances directly so it works regardless of host module
    (the _SwiGLUMortar hidden-neuron router).
    """
    stats: Dict[str, float] = {}
    for name, m in model.named_modules():
        if isinstance(m, TileRouter):
            rs = m.log_stats()
            safe = name.replace(".", "_")
            for k, v in rs.items():
                stats[f"{k}/{safe}"] = v
    return stats
