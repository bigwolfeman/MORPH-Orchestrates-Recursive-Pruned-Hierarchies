"""
Neural Long-Term Memory module from Titans paper (arxiv 2501.00663).

PAPER-FAITHFUL v3: Learned data-dependent gates, SiLU + L2-norm projections,
chunk-level processing. Fixes identified via cross-referencing paper with
lucidrains, Aedelon, and kolejnyy implementations.

Key equations from Section 3.1:
    - Associative Loss: l(M; x) = ||M(k) - v||^2  (Eq. 12)
    - Momentum: S_t = η_t * S_{t-1} - θ_t * ∇l(M; x)  (Eq. 13)
    - Memory Update: M_t = (1 - α_t) * M_{t-1} + S_t  (Eq. 14)
    - Retrieve: y_t = M*(q_t) with stop-grad on M weights  (Eq. 15)

Critical fixes over v2:
    - α_t, η_t, θ_t are LEARNED sigmoid(Linear(x_chunk)), not fixed constants
    - SiLU activation + L2-normalization on key/query projections (Section 4.4)
    - Chunk-level processing (chunk_size=128, matching HCA compress ratio)
    - No bias on projections (paper convention)
    - Functional probe on last chunk gives gate networks gradient signal

Version: 3.0.0
"""

from typing import Dict, List, Optional, Tuple

import math
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .norms import RMSNorm


# ── Legacy stubs (checkpoint compat, not used) ──────────────────────────────

class ForgetGate(nn.Module):
    """LEGACY: Kept for import compatibility only."""
    def __init__(self, d_model: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class DecayGate(nn.Module):
    """LEGACY: Kept for import compatibility only."""
    def __init__(self, d_model: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Memory MLP ──────────────────────────────────────────────────────────────

class DeepMemoryMLP(nn.Module):
    """Deep MLP that serves as the mutable memory delta.

    Split-path design: the MLP is a pure delta branch (NO residual).
    The stable read path (RMSNorm(q)) lives in NeuralMemory.retrieve().

    forward(x) returns ONLY the MLP output, normalized. The caller
    combines it with the stable path via a learned delta gate.

    Args:
        d_model: Input/output dimension.
        n_layers: Number of MLP layers (paper recommends >= 2).
        hidden_dim: Hidden dimension (defaults to d_model).
        activation: Activation function.
        use_bias: Whether linear layers use bias (paper: False).
        zero_init_output: Zero-init the final layer so delta starts at 0.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
        use_bias: bool = False,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim or d_model

        act_fn = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = []
        if n_layers == 1:
            layers.append(nn.Linear(d_model, d_model, bias=use_bias))
        else:
            layers.append(nn.Linear(d_model, self.hidden_dim, bias=use_bias))
            layers.append(act_fn())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias))
                layers.append(act_fn())
            layers.append(nn.Linear(self.hidden_dim, d_model, bias=use_bias))

        self.mlp = nn.Sequential(*layers)
        self.output_norm = RMSNorm(d_model)
        self._init_weights(zero_init_output)

    def _init_weights(self, zero_init_output: bool = True):
        for i, m in enumerate(self.mlp.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if zero_init_output:
            last_linear = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.mlp(x))


# ── Neural Memory (Paper-Faithful v3) ──────────────────────────────────────

class NeuralMemory(nn.Module):
    """Neural Long-Term Memory with learned data-dependent gates.

    Paper-faithful implementation with chunk-level processing.  The three
    gate scalars (α forget, η momentum, θ learning-rate) are produced by
    small linear layers applied to the chunk-pooled representation, matching
    the paper's "parameters as functions of chunks" (Section 3.2).

    Args:
        d_model: Input dimension from model hidden states.
        d_memory: Memory MLP dimension (default: d_model).
        n_memory_layers: Depth of memory MLP (>= 2 recommended).
        chunk_size: Tokens per chunk for gate computation (128 = HCA block size).
        max_lr: Upper bound for learned θ_t (default 1.0, paper convention).
        gate_probe_weight: Weight of probe loss for gate learning (0.05 default).
    """

    def __init__(
        self,
        d_model: int,
        d_memory: Optional[int] = None,
        n_memory_layers: int = 2,
        chunk_size: int = 128,
        max_lr: float = 1.0,
        gate_probe_weight: float = 0.05,
        # Legacy kwargs accepted but ignored
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_memory = d_memory or d_model
        self.n_memory_layers = n_memory_layers
        self.chunk_size = chunk_size
        self.max_lr = max_lr
        self.gate_probe_weight = gate_probe_weight

        # ── Projections: SiLU + L2-norm (Section 4.4), no bias ──────────
        self.W_K = nn.Linear(d_model, self.d_memory, bias=False)
        self.W_V = nn.Linear(d_model, self.d_memory, bias=False)
        self.W_Q = nn.Linear(d_model, self.d_memory, bias=False)

        for proj in [self.W_K, self.W_V, self.W_Q]:
            nn.init.xavier_uniform_(proj.weight)

        # ── Stable read path: RMSNorm(q) — always active ────────────────
        self.stable_norm = RMSNorm(self.d_memory)

        # ── Mutable delta: MLP(q), zero-init output, NO residual ────────
        self.memory_mlp = DeepMemoryMLP(
            d_model=self.d_memory,
            n_layers=n_memory_layers,
            use_bias=False,
            zero_init_output=True,
        )

        # ── Delta gate: controls mutable branch contribution ────────────
        # Starts near 0 (log-scale init), learned by outer optimizer.
        # retrieve(q) = stable(q) + sigmoid(delta_gate_raw) * delta(q)
        self.delta_gate_raw = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007

        # ── Learned per-chunk gates ─────────────────────────────────────
        # sigmoid(Linear(chunk_pooled)) → scalar per chunk.
        # Bias initialized so sigmoid(bias) ≈ target default value.
        self.alpha_gate = nn.Linear(self.d_memory, 1, bias=True)
        self.eta_gate = nn.Linear(self.d_memory, 1, bias=True)
        self.theta_gate = nn.Linear(self.d_memory, 1, bias=True)

        nn.init.zeros_(self.alpha_gate.weight)
        nn.init.constant_(self.alpha_gate.bias, -4.6)  # sigmoid(-4.6) ≈ 0.01

        nn.init.zeros_(self.eta_gate.weight)
        nn.init.constant_(self.eta_gate.bias, 0.4)     # sigmoid(0.4) ≈ 0.60

        nn.init.zeros_(self.theta_gate.weight)
        nn.init.constant_(self.theta_gate.bias, -2.2)  # sigmoid(-2.2)*1.0 ≈ 0.10

        # ── Momentum buffer ─────────────────────────────────────────────
        n_params = sum(p.numel() for p in self.memory_mlp.mlp.parameters())
        self.register_buffer("momentum_S", torch.zeros(n_params))

        # Pre-computed parameter offsets for fast flatten/unflatten
        self._param_offsets: list[tuple[int, tuple[int, ...]]] = []
        offset = 0
        for p in self.memory_mlp.mlp.parameters():
            self._param_offsets.append((offset, p.shape))
            offset += p.numel()
        self._n_mlp_params = n_params

        # Last update stats for logging
        self._last_stats: Optional[Dict[str, float]] = None

    # ── Projections (Paper Section 4.4) ─────────────────────────────────

    def _project_k(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU activation + L2-normalize keys."""
        return F.normalize(F.silu(self.W_K(x)), dim=-1)

    def _project_v(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU activation on values (no L2-norm — preserve magnitude info)."""
        return F.silu(self.W_V(x))

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU activation + L2-normalize queries."""
        return F.normalize(F.silu(self.W_Q(x)), dim=-1)

    # ── Chunk-level gate computation ────────────────────────────────────

    def _compute_chunk_gates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Compute per-chunk α, η, θ from mean-pooled chunk representations.

        Args:
            x: [B, T, D] input (in model space, NOT projected).

        Returns:
            alpha: [B, nc, 1] forget gate (small = retain).
            eta:   [B, nc, 1] momentum coefficient.
            theta: [B, nc, 1] learning rate (scaled by max_lr).
            nc:    number of chunks.
        """
        B, T, D = x.shape
        cs = self.chunk_size
        nc = max(1, T // cs)
        actual_cs = T // nc  # handle T < chunk_size

        x_chunked = x[:, :nc * actual_cs].reshape(B, nc, actual_cs, D)
        x_pool = x_chunked.mean(dim=2)  # [B, nc, D]

        # Project into memory space for gate input (shared representation)
        k_pool = F.normalize(F.silu(self.W_K(x_pool)), dim=-1)  # [B, nc, d_memory]

        alpha = self.alpha_gate(k_pool).sigmoid()                  # [B, nc, 1]
        eta = self.eta_gate(k_pool).sigmoid()                      # [B, nc, 1]
        theta = self.theta_gate(k_pool).sigmoid() * self.max_lr   # [B, nc, 1]
        return alpha, eta, theta, nc

    # ── Flatten / unflatten helpers ─────────────────────────────────────

    def _flatten_mlp_params(self) -> torch.Tensor:
        """Flatten MLP parameters into a single vector."""
        return torch.cat([p.view(-1) for p in self.memory_mlp.mlp.parameters()])

    def _unflatten_to_dict(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert flat vector to named param dict for functional_call on memory_mlp.

        Only overrides mlp.* weights (updated by inner GD). output_norm.*
        keeps its current values automatically via functional_call semantics.
        """
        param_dict = {}
        offset = 0
        for name, p in self.memory_mlp.mlp.named_parameters():
            numel = p.numel()
            param_dict["mlp." + name] = flat[offset:offset + numel].view(p.shape)
            offset += numel
        return param_dict

    def _set_mlp_params_from_flat(self, flat: torch.Tensor):
        """Copy flat vector into MLP parameters (in-place, no grad)."""
        with torch.no_grad():
            for (off, shape), p in zip(self._param_offsets, self.memory_mlp.mlp.parameters()):
                p.data.copy_(flat[off:off + p.numel()].view(shape))

    # ── Core update ─────────────────────────────────────────────────────

    @torch._dynamo.disable
    def update(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
        z_target: Optional[torch.Tensor] = None,
        z_head: Optional[nn.Module] = None,
        suppress_decay: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Update memory MLP with per-chunk learned gates.

        Processes the sequence in chunks.  Chunks 0..n-2 are updated with
        standard (non-differentiable) inner GD.  The LAST chunk uses a
        functional probe: gate values stay in the autograd graph so the
        outer optimizer can train the gate networks via the probe loss.

        Args:
            x: [B, T, D] input hidden states.
            return_stats: Return a stats dict instead of scalar loss.
            z_target: Optional [B, T, d_z] JEPA target (detached).
            z_head: Optional Linear mapping d_memory → d_z.
            suppress_decay: Force alpha=0 (warmup mode).

        Returns:
            If return_stats=False: probe_loss (differentiable w.r.t. gate nets).
            If return_stats=True: dict with loss, gate values, grad_norm.
        """
        B, T, D = x.shape
        cs = self.chunk_size
        nc = max(1, T // cs)
        actual_cs = T // nc

        # Compute per-chunk gates (differentiable)
        alpha_all, eta_all, theta_all, nc = self._compute_chunk_gates(x)

        if suppress_decay:
            alpha_all = torch.zeros_like(alpha_all)

        total_loss = 0.0
        last_grad_norm = 0.0

        # ── Process chunks 0..nc-2: non-differentiable fast path ────────
        for i in range(nc - 1):
            chunk_start = i * actual_cs
            chunk_end = chunk_start + actual_cs
            x_chunk = x[:, chunk_start:chunk_end]

            with torch.enable_grad():
                loss_i = self._chunk_loss(x_chunk, z_target, z_head, chunk_start, chunk_end)
                grads = torch.autograd.grad(
                    loss_i, self.memory_mlp.mlp.parameters(),
                    retain_graph=False, create_graph=False,
                )

            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            gn = flat_grad.norm()
            if gn > 1.0:
                flat_grad = flat_grad / gn
            if torch.isnan(flat_grad).any():
                total_loss += loss_i.detach().item()
                continue

            a = alpha_all[:, i, 0].mean().item()
            e = eta_all[:, i, 0].mean().item()
            t = theta_all[:, i, 0].mean().item()

            with torch.no_grad():
                self.momentum_S.mul_(e).add_(flat_grad, alpha=-t)
                for (off, shape), p in zip(self._param_offsets, self.memory_mlp.mlp.parameters()):
                    n = p.numel()
                    p.data.mul_(1.0 - a).add_(self.momentum_S[off:off + n].view(shape))

            total_loss += loss_i.detach().item()
            last_grad_norm = gn.item()

        # ── Last chunk: functional probe for gate learning ──────────────
        last_start = (nc - 1) * actual_cs
        last_end = last_start + actual_cs
        x_last = x[:, last_start:last_end]

        with torch.enable_grad():
            loss_last = self._chunk_loss(x_last, z_target, z_head, last_start, last_end)
            grads_last = torch.autograd.grad(
                loss_last, self.memory_mlp.mlp.parameters(),
                retain_graph=False, create_graph=False,
            )

        flat_grad_last = torch.cat([g.reshape(-1) for g in grads_last]).detach()
        gn_last = flat_grad_last.norm()
        if gn_last > 1.0:
            flat_grad_last = flat_grad_last / gn_last
        last_grad_norm = gn_last.item()

        # Gate values for last chunk — KEEP IN GRAPH for probe gradient
        a_t = alpha_all[:, -1, 0].mean()   # scalar, differentiable
        e_t = eta_all[:, -1, 0].mean()
        t_t = theta_all[:, -1, 0].mean()

        # Functional update: new_params = (1-α)*old + η*momentum - θ*grad
        # .clone() required: detach() shares storage, in-place update later would corrupt backward graph
        flat_params = self._flatten_mlp_params().detach()
        momentum_detached = self.momentum_S.detach().clone()
        new_S = e_t * momentum_detached - t_t * flat_grad_last
        new_params_flat = (1.0 - a_t) * flat_params + new_S
        new_params_dict = self._unflatten_to_dict(new_params_flat)

        # Probe: evaluate post-update memory on same chunk (detach inputs
        # so only gate networks get gradients from probe)
        k_probe = self._project_k(x_last).detach()
        probe_predicted = functional_call(self.memory_mlp, new_params_dict, (k_probe,))
        if z_target is not None and z_head is not None:
            z_slice = z_target[:, last_start:last_end].detach()
            z_pred = z_head(probe_predicted)
            probe_loss = F.smooth_l1_loss(z_pred, z_slice)
            probe_loss = probe_loss + 0.01 * torch.relu(
                1.0 - z_pred.std(dim=(0, 1)).clamp(min=1e-4)
            ).mean()
        else:
            v_probe = self._project_v(x_last).detach()
            probe_loss = F.mse_loss(probe_predicted, v_probe)

        # Now do the ACTUAL in-place update for the last chunk
        with torch.no_grad():
            a_val = a_t.item()
            e_val = e_t.item()
            t_val = t_t.item()
            self.momentum_S.mul_(e_val).add_(flat_grad_last, alpha=-t_val)
            for (off, shape), p in zip(self._param_offsets, self.memory_mlp.mlp.parameters()):
                n = p.numel()
                p.data.mul_(1.0 - a_val).add_(self.momentum_S[off:off + n].view(shape))

        total_loss += loss_last.detach().item()
        avg_loss = total_loss / nc

        # Store stats for logging
        self._last_stats = {
            "alpha_t": a_val if nc > 0 else 0.0,
            "eta_t": e_val if nc > 0 else 0.0,
            "theta_t": t_val if nc > 0 else 0.0,
            "grad_norm": last_grad_norm,
            "avg_chunk_loss": avg_loss,
            "n_chunks": nc,
        }

        if return_stats:
            stats = dict(self._last_stats)
            stats["loss"] = probe_loss
            stats["probe_loss"] = probe_loss
            return stats

        return probe_loss

    def _chunk_loss(
        self,
        x_chunk: torch.Tensor,
        z_target: Optional[torch.Tensor],
        z_head: Optional[nn.Module],
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Compute memory loss on a single chunk."""
        k = self._project_k(x_chunk)
        predicted = self.memory_mlp(k)

        if z_target is not None and z_head is not None:
            z_pred = z_head(predicted)
            z_slice = z_target[:, start:end].detach()
            loss = F.smooth_l1_loss(z_pred, z_slice)
            loss = loss + 0.01 * torch.relu(
                1.0 - z_pred.std(dim=(0, 1)).clamp(min=1e-4)
            ).mean()
        else:
            v = self._project_v(x_chunk).detach()
            loss = F.mse_loss(predicted, v)
        return loss

    # ── Retrieve (split path) ──────────────────────────────────────────

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        """Split-path retrieve: stable projection + gated mutable delta.

        out = stable_norm(q) + sigmoid(delta_gate) * memory_mlp(q)

        Stable path: always active, trained by outer optimizer only.
        Delta path: MLP updated by inner GD, zero-init, gated near 0.

        Diagnostic flags:
            _zero_output: return zeros (full path disabled)
            _zero_delta: return stable only (disable mutable branch)
            _zero_stable: return delta only (disable stable branch)
        """
        q = self._project_q(x)

        if getattr(self, '_zero_output', False):
            return torch.zeros_like(q)

        stable = self.stable_norm(q)

        if getattr(self, '_zero_delta', False):
            return stable

        if getattr(self, '_zero_stable', False):
            stable = torch.zeros_like(stable)

        # Mutable delta: stop-grad on MLP weights, STE for W_Q gradient
        with torch.no_grad():
            delta = self.memory_mlp(q)
        if q.requires_grad:
            delta = delta + (q - q.detach())

        g = torch.sigmoid(self.delta_gate_raw)
        return stable + g * delta

    # ── Utilities ───────────────────────────────────────────────────────

    def reset_momentum(self):
        """Reset momentum buffers between sequences."""
        with torch.no_grad():
            self.momentum_S.zero_()

    def reset_memory(self):
        """Full reset: re-initialize MLP and momentum."""
        with torch.no_grad():
            self.memory_mlp._init_weights()
            self.momentum_S.zero_()

    def get_memory_stats(self) -> Dict[str, float]:
        """Statistics for wandb logging."""
        with torch.no_grad():
            flat = self._flatten_mlp_params()
            param_norm = flat.norm().item()
            momentum_norm = self.momentum_S.norm().item()
            delta_gate = torch.sigmoid(self.delta_gate_raw).item()

        stats = {
            "memory_param_norm": param_norm,
            "momentum_norm": momentum_norm,
            "delta_gate": delta_gate,
            "n_memory_layers": float(self.n_memory_layers),
            "d_memory": float(self.d_memory),
            "n_params": float(self._n_mlp_params),
        }
        if self._last_stats is not None:
            stats.update({
                "alpha_t": self._last_stats["alpha_t"],
                "eta_t": self._last_stats["eta_t"],
                "theta_t": self._last_stats["theta_t"],
                "grad_norm": self._last_stats["grad_norm"],
            })
        return stats

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_memory={self.d_memory}, "
            f"n_layers={self.n_memory_layers}, chunk_size={self.chunk_size}, "
            f"max_lr={self.max_lr}"
        )
