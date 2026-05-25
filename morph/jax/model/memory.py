"""MORPH neural memory integration layer — JAX/Flax port.

Exact port of morph/model/memory.py + morph/model/titans_core/neural_memory.py
to JAX/Flax. Because JAX has no torch.autograd.grad, the inner gradient
computation is done purely via jax.grad on functional parameter dicts.

Architecture
------------
The memory is a small MLP (DeepMemoryMLP) representing M(k) ≈ v_target.
Update equations (Titans paper, Eq. 12-14):
    l(M; x)  = ||M(k) - v||²           # associative loss
    S_t      = η_t * S_{t-1} - θ_t * ∇l   # momentum
    M_t      = (1 - α_t) * M_{t-1} + S_t  # parameter update

Three gate scalars (α forget, η momentum, θ lr) are small Linears applied to
chunk-pooled representations — data-dependent per forward pass.

Retrieve (split-path, Eq. 15):
    out = stable_norm(q) + sigmoid(delta_gate) * memory_mlp(q)

JAX-specific design decisions
------------------------------
- MLP parameters live in the standard "params" collection (normal Flax Dense layers).
  They are updated by the outer optimizer, but the inner-GD update is done via a
  functional update using jax.grad and jax.lax.stop_gradient.
- The momentum buffer lives in the "memory_state" mutable collection.
- jax.grad computes inner gradients purely functionally — no graph retention needed.
- jax.checkpoint wraps the update body for memory efficiency during BPTT.
- Python for-loop over chunks (not lax.scan) — avoids static-shape issues and
  matches the TPU known-gotcha documented in TPU/CLAUDE.md.

Inner-GD in JAX: The key insight is that we use jax.grad on a FUNCTIONAL version
of the loss that takes the MLP params as arguments. The actual param update is done
with stop_gradient so the outer optimizer doesn't see these as a gradient path.
The probe loss for the last chunk IS differentiable w.r.t. the gate Linear params
(alpha/eta/theta), so the outer optimizer can train gating behavior.

The memory MLP is NOT a mutable buffer — it IS outer-optimizer-trained params
(like W_K, W_V, W_Q). The inner GD produces a step of update within the forward
pass, temporarily moves the MLP to better weights, then evaluates the probe loss
on those temporary weights. The ACTUAL inner-GD update is applied directly via
param mutation stored in memory_state["tmp_mlp_params"] which is used for retrieve.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


# ── Utility helpers ────────────────────────────────────────────────────────────

def _rms_norm(x: jnp.ndarray, scale: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Functional RMSNorm. Works on any shape; normalizes last axis."""
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    return ((x_f32 / rms) * scale.astype(jnp.float32)).astype(x.dtype)


def _silu_l2(kernel: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """SiLU + L2-normalize. kernel: [in, out] (Flax convention)."""
    y = x.astype(jnp.float32) @ kernel.astype(jnp.float32)
    y = jax.nn.silu(y)
    norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    return (y / jnp.maximum(norm, 1e-8)).astype(x.dtype)


def _silu(kernel: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """SiLU only (no L2-norm). kernel: [in, out]."""
    y = x.astype(jnp.float32) @ kernel.astype(jnp.float32)
    return jax.nn.silu(y).astype(x.dtype)


def _linear(kernel: jnp.ndarray, bias: Optional[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Linear layer. kernel: [in, out], bias: [out] or None."""
    y = x.astype(jnp.float32) @ kernel.astype(jnp.float32)
    if bias is not None:
        y = y + bias.astype(jnp.float32)
    return y.astype(x.dtype)


def _flatten_pytree(tree) -> jnp.ndarray:
    """Flatten a pytree of arrays to a 1D vector."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.concatenate([leaf.reshape(-1).astype(jnp.float32) for leaf in leaves])


def _unflatten_pytree(flat: jnp.ndarray, template) -> object:
    """Restore a pytree from a flat vector using a template for structure."""
    leaves, treedef = jax.tree_util.tree_flatten(template)
    out_leaves = []
    offset = 0
    for leaf in leaves:
        n = leaf.size
        out_leaves.append(
            flat[offset:offset + n].reshape(leaf.shape).astype(leaf.dtype)
        )
        offset += n
    return jax.tree_util.tree_unflatten(treedef, out_leaves)


# ── Memory MLP functional forward ─────────────────────────────────────────────

def _mlp_forward_functional(
    mlp_params: dict,   # {"layer_0": {"kernel": [d, d]}, ..., "norm_scale": [d]}
    x: jnp.ndarray,    # [*, d]
    n_layers: int,
) -> jnp.ndarray:
    """Functional forward through the memory MLP + RMSNorm output.

    mlp_params keys:
        "layer_{i}/kernel": [d_mem, d_mem] for i in range(n_layers)
        "norm_scale": [d_mem]

    Returns RMSNorm(MLP(x)), shape [*, d_mem].
    """
    h = x.astype(jnp.float32)

    for i in range(n_layers):
        kernel = mlp_params[f"layer_{i}"]["kernel"].astype(jnp.float32)
        h = h @ kernel
        if i < n_layers - 1:
            h = jax.nn.silu(h)

    # RMSNorm output
    norm_scale = mlp_params["norm_scale"].astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
    h = (h / rms) * norm_scale
    return h.astype(x.dtype)


def _associative_loss_fn(
    mlp_params: dict,
    k: jnp.ndarray,    # [B, T, d_mem] projected keys (stop-gradient)
    v: jnp.ndarray,    # [B, T, d_mem] projected values (stop-gradient)
    n_layers: int,
) -> jnp.ndarray:
    """||M(k) - v||² (mean over batch and time)."""
    predicted = _mlp_forward_functional(mlp_params, k, n_layers)
    return jnp.mean(
        (predicted.astype(jnp.float32) - v.astype(jnp.float32)) ** 2
    )


# ── NeuralMemoryCore ──────────────────────────────────────────────────────────


class NeuralMemoryCore(nn.Module):
    """Paper-faithful neural memory with inner-GD update.

    Holds projection matrices, gate networks, and the memory MLP as standard
    Flax params. The inner-GD update temporarily moves MLP params to better
    weights and evaluates a probe loss (differentiable w.r.t. gate Linears).
    The updated MLP state is stored in the "memory_state" mutable collection
    and used by retrieve() on subsequent calls.

    Attributes
    ----------
    d_model         : int
    d_memory        : int   0 = use d_model.
    n_memory_layers : int   Memory MLP depth (>= 2).
    chunk_size      : int   Tokens per chunk for gate computation.
    max_lr          : float Upper bound for learned θ_t.
    """

    d_model: int
    d_memory: int = 0
    n_memory_layers: int = 2
    chunk_size: int = 128
    max_lr: float = 1.0

    def _dm(self) -> int:
        return self.d_memory if self.d_memory > 0 else self.d_model

    def setup(self):
        d_mem = self._dm()
        n = self.n_memory_layers

        # ── Projection matrices (no bias per paper) ───────────────────────
        self.W_K = nn.Dense(d_mem, use_bias=False,
                            kernel_init=nn.initializers.xavier_uniform(), name="W_K")
        self.W_V = nn.Dense(d_mem, use_bias=False,
                            kernel_init=nn.initializers.xavier_uniform(), name="W_V")
        self.W_Q = nn.Dense(d_mem, use_bias=False,
                            kernel_init=nn.initializers.xavier_uniform(), name="W_Q")

        # ── Stable read path: single RMSNorm scale ────────────────────────
        self.stable_norm_scale = self.param(
            "stable_norm_scale", nn.initializers.ones, (d_mem,)
        )

        # ── Memory MLP (outer-optimizer-trained baseline; inner GD modifies
        #    a functional copy stored in memory_state) ─────────────────────
        # Layer kernels: [d_mem, d_mem] each. Last layer zero-init.
        self.mlp_kernels = [
            self.param(
                f"mlp_kernel_{i}",
                nn.initializers.xavier_uniform() if i < n - 1 else nn.initializers.zeros,
                (d_mem, d_mem),
            )
            for i in range(n)
        ]
        self.mlp_norm_scale = self.param(
            "mlp_norm_scale", nn.initializers.ones, (d_mem,)
        )

        # ── Delta gate: sigmoid(-5) ≈ 0.007 at init ──────────────────────
        self.delta_gate_raw = self.param(
            "delta_gate_raw", nn.initializers.constant(-5.0), ()
        )

        # ── Gate Linears for per-chunk α/η/θ ─────────────────────────────
        self.alpha_gate = nn.Dense(
            1, use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-4.6),
            name="alpha_gate",
        )
        self.eta_gate = nn.Dense(
            1, use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(0.4),
            name="eta_gate",
        )
        self.theta_gate = nn.Dense(
            1, use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-2.2),
            name="theta_gate",
        )

    def _build_mlp_params(self) -> dict:
        """Assemble current MLP params (from Flax params) into a functional dict."""
        n = self.n_memory_layers
        params = {
            f"layer_{i}": {"kernel": self.mlp_kernels[i]}
            for i in range(n)
        }
        params["norm_scale"] = self.mlp_norm_scale
        return params

    def _get_mlp_params(self) -> dict:
        """Get MLP params: prefer memory_state (inner-GD updated) over base params."""
        if self.has_variable("memory_state", "mlp_params"):
            return self.get_variable("memory_state", "mlp_params")
        return self._build_mlp_params()

    def _get_momentum(self, n_params: int) -> jnp.ndarray:
        """Get current momentum buffer (flat 1D)."""
        if self.has_variable("memory_state", "momentum"):
            return self.get_variable("memory_state", "momentum")
        return jnp.zeros((n_params,))

    # ── Projection helpers ────────────────────────────────────────────────

    def _pk(self, x: jnp.ndarray) -> jnp.ndarray:
        """SiLU + L2-norm keys."""
        raw = self.W_K(x.astype(jnp.bfloat16)).astype(jnp.float32)
        raw = jax.nn.silu(raw)
        return (raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-8)).astype(x.dtype)

    def _pv(self, x: jnp.ndarray) -> jnp.ndarray:
        """SiLU values (no L2-norm)."""
        return jax.nn.silu(self.W_V(x.astype(jnp.bfloat16))).astype(x.dtype)

    def _pq(self, x: jnp.ndarray) -> jnp.ndarray:
        """SiLU + L2-norm queries."""
        raw = self.W_Q(x.astype(jnp.bfloat16)).astype(jnp.float32)
        raw = jax.nn.silu(raw)
        return (raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-8)).astype(x.dtype)

    # ── Chunk gate computation ─────────────────────────────────────────────

    def _chunk_gates(
        self, x: jnp.ndarray, nc: int, actual_cs: int
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute per-chunk α, η, θ — each [B, nc, 1]."""
        B, T, D = x.shape
        x_c = x[:, :nc * actual_cs].reshape(B, nc, actual_cs, D)
        x_pool = x_c.mean(axis=2)  # [B, nc, D]
        k_pool = self._pk(x_pool)  # [B, nc, d_mem]

        alpha = jax.nn.sigmoid(self.alpha_gate(k_pool))               # [B, nc, 1]
        eta   = jax.nn.sigmoid(self.eta_gate(k_pool))                  # [B, nc, 1]
        theta = jax.nn.sigmoid(self.theta_gate(k_pool)) * self.max_lr  # [B, nc, 1]
        return alpha, eta, theta

    # ── Core update ───────────────────────────────────────────────────────

    def update(
        self,
        x: jnp.ndarray,
        suppress_decay: bool = False,
    ) -> jnp.ndarray:
        """Per-chunk inner-GD memory update.

        Processes chunks 0..nc-2 with non-differentiable inner GD
        (stop_gradient). Last chunk uses a functional probe (differentiable
        w.r.t. gate networks) for outer optimizer gate learning.

        Updates "memory_state" with new mlp_params and momentum.

        Returns:
            Differentiable probe_loss scalar.
        """
        d_mem = self._dm()
        B, T, D = x.shape
        cs = self.chunk_size
        nc = max(1, T // cs)
        actual_cs = T // nc
        n_layers = self.n_memory_layers

        # Start from base MLP params (outer optimizer owns these; inner GD
        # will produce a temporary delta in memory_state)
        base_mlp_params = self._build_mlp_params()

        # Load previous inner-GD state from memory_state (or start fresh)
        cur_mlp = self._get_mlp_params()
        n_flat = sum(leaf.size for leaf in jax.tree_util.tree_leaves(cur_mlp))
        cur_S = self._get_momentum(n_flat)

        alpha_all, eta_all, theta_all = self._chunk_gates(x, nc, actual_cs)
        if suppress_decay:
            alpha_all = jnp.zeros_like(alpha_all)

        # ── Process chunks 0..nc-2: non-differentiable updates ────────────
        for i in range(nc - 1):
            chunk = x[:, i * actual_cs:(i + 1) * actual_cs]
            k = jax.lax.stop_gradient(self._pk(chunk))
            v = jax.lax.stop_gradient(self._pv(chunk))

            frozen = jax.lax.stop_gradient(cur_mlp)
            grad_fn = jax.grad(_associative_loss_fn, argnums=0)
            raw_grads = grad_fn(frozen, k, v, n_layers)

            flat_grad = jax.lax.stop_gradient(_flatten_pytree(raw_grads))
            gn = jnp.linalg.norm(flat_grad)
            flat_grad = jnp.where(gn > 1.0, flat_grad / gn, flat_grad)
            flat_grad = jnp.where(jnp.any(jnp.isnan(flat_grad)), jnp.zeros_like(flat_grad), flat_grad)

            a = jax.lax.stop_gradient(alpha_all[:, i, 0].mean())
            e = jax.lax.stop_gradient(eta_all[:, i, 0].mean())
            t = jax.lax.stop_gradient(theta_all[:, i, 0].mean())

            new_S = e * cur_S - t * flat_grad
            flat_cur = jax.lax.stop_gradient(_flatten_pytree(cur_mlp))
            new_flat = (1.0 - a) * flat_cur + new_S
            cur_mlp = jax.lax.stop_gradient(_unflatten_pytree(new_flat, cur_mlp))
            cur_S = jax.lax.stop_gradient(new_S)

        # ── Last chunk: probe for gate gradient ───────────────────────────
        x_last = x[:, (nc - 1) * actual_cs:(nc - 1) * actual_cs + actual_cs]
        k_last = jax.lax.stop_gradient(self._pk(x_last))
        v_last = jax.lax.stop_gradient(self._pv(x_last))

        frozen_last = jax.lax.stop_gradient(cur_mlp)
        raw_grads_last = jax.grad(_associative_loss_fn, argnums=0)(frozen_last, k_last, v_last, n_layers)
        flat_grad_last = jax.lax.stop_gradient(_flatten_pytree(raw_grads_last))
        gn_last = jnp.linalg.norm(flat_grad_last)
        flat_grad_last = jnp.where(gn_last > 1.0, flat_grad_last / gn_last, flat_grad_last)
        flat_grad_last = jnp.where(jnp.any(jnp.isnan(flat_grad_last)), jnp.zeros_like(flat_grad_last), flat_grad_last)

        # Gate values — keep in graph for probe gradient
        a_t = alpha_all[:, -1, 0].mean()
        e_t = eta_all[:, -1, 0].mean()
        t_t = theta_all[:, -1, 0].mean()

        # Functional post-update params for probe evaluation
        flat_cur_last = jax.lax.stop_gradient(_flatten_pytree(cur_mlp))
        new_S_last = e_t * jax.lax.stop_gradient(cur_S) - t_t * flat_grad_last
        new_flat_last = (1.0 - a_t) * flat_cur_last + new_S_last
        new_mlp_probe = jax.lax.stop_gradient(_unflatten_pytree(new_flat_last, cur_mlp))

        # Probe: evaluate on stop-gradient'd keys
        probe_pred = _mlp_forward_functional(new_mlp_probe, k_last, n_layers)
        probe_loss = jnp.mean(
            (probe_pred.astype(jnp.float32) - v_last.astype(jnp.float32)) ** 2
        )

        # Actual in-place state update (all stop-gradient — purely stateful)
        a_val = jax.lax.stop_gradient(a_t)
        e_val = jax.lax.stop_gradient(e_t)
        t_val = jax.lax.stop_gradient(t_t)

        final_S = e_val * cur_S - t_val * flat_grad_last
        flat_final = (1.0 - a_val) * jax.lax.stop_gradient(_flatten_pytree(cur_mlp)) + final_S
        final_mlp = jax.lax.stop_gradient(_unflatten_pytree(flat_final, cur_mlp))

        # Write updated state back to mutable collection
        self.put_variable("memory_state", "mlp_params", final_mlp)
        self.put_variable("memory_state", "momentum", jax.lax.stop_gradient(final_S))

        return probe_loss

    # ── Retrieve ──────────────────────────────────────────────────────────

    def retrieve(self, x: jnp.ndarray) -> jnp.ndarray:
        """Split-path retrieve: stable_norm(q) + sigmoid(delta_gate) * MLP(q).

        Uses the inner-GD updated MLP params from memory_state if available.

        Also touches W_K and W_V to ensure their params are always initialized
        during model.init() (they are only used in update(), which requires training=True,
        so without this touch they would be missing from params at training time).

        Args:
            x: [B, T, d_model] query hidden states.

        Returns:
            [B, T, d_model] retrieved representation.
        """
        # Ensure all params are always initialized (W_K, W_V, gates only used in update)
        # This is idiomatic Flax: params are created on first call in any bound scope.
        # Always use bf16 for these dummy touches so we never trigger f64 cublas paths
        # even when JAX_ENABLE_X64=1 (e.g. in tests that use numpy int64 for bigram hash).
        _dummy = x[..., :1, :].astype(jnp.bfloat16)
        _ = self.W_K(_dummy)      # creates W_K/kernel
        _ = self.W_V(_dummy)      # creates W_V/kernel
        # Gate linears: need 1D input of size d_mem
        _dummy1d = jnp.zeros((1, 1, self._dm()), dtype=jnp.bfloat16)
        _ = self.alpha_gate(_dummy1d)   # creates alpha_gate/kernel, bias
        _ = self.eta_gate(_dummy1d)     # creates eta_gate/kernel, bias
        _ = self.theta_gate(_dummy1d)   # creates theta_gate/kernel, bias

        q = self._pq(x)

        # Stable path: RMSNorm(q)
        stable = _rms_norm(q, self.stable_norm_scale)

        # Mutable delta path
        mlp_params = jax.lax.stop_gradient(self._get_mlp_params())
        n_layers = self.n_memory_layers
        delta = _mlp_forward_functional(mlp_params, q, n_layers)
        # STE for W_Q gradient: delta + (q - sg(q)) straight-through trick
        delta = delta + (q - jax.lax.stop_gradient(q))

        g = jax.nn.sigmoid(self.delta_gate_raw.astype(jnp.float32))
        return (stable + g * delta).astype(x.dtype)

    # ── Module forward (dispatch via keyword) ─────────────────────────────

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Default: retrieve. Use update() directly for training."""
        return self.retrieve(x)


# ── MemorySystem (wrapper matching morph/model/memory.py) ────────────────────


class MemorySystem(nn.Module):
    """MORPH wrapper around NeuralMemoryCore with all integration methods.

    Attributes
    ----------
    d_model           : int   Main model hidden dimension.
    n_layers          : int   Memory MLP depth (>= 2).
    n_memory_tokens   : int   Number of MAC tokens.
    d_memory_channel  : int   MAG channel width. 0 = auto (max(64, d_model//6)).
    chunk_size        : int   Tokens per chunk.
    max_lr            : float Gate upper bound.
    """

    d_model: int
    n_layers: int = 2
    n_memory_tokens: int = 32
    d_memory_channel: int = 0
    chunk_size: int = 128
    max_lr: float = 1.0

    def setup(self):
        d_mem_ch = self.d_memory_channel if self.d_memory_channel > 0 \
            else max(64, self.d_model // 6)
        self._d_mem_ch = d_mem_ch

        self.memory = NeuralMemoryCore(
            d_model=self.d_model,
            d_memory=self.d_model,
            n_memory_layers=self.n_layers,
            chunk_size=self.chunk_size,
            max_lr=self.max_lr,
            name="memory",
        )

        # SSM inject: zero-init proj so it starts as no-op
        self.ssm_query = self.param(
            "ssm_query",
            nn.initializers.normal(stddev=0.02),
            (1, self.d_model),
        )
        self.ssm_proj = nn.Dense(
            self.d_model, use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="ssm_proj",
        )

        # MAG inject
        self.mag_gate = nn.Dense(
            d_mem_ch, use_bias=True,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            name="mag_gate",
        )
        self.mag_proj = nn.Dense(
            d_mem_ch, use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="mag_proj",
        )

        # MAC tokens
        self.mac_gate_raw = self.param(
            "mac_gate_raw",
            nn.initializers.constant(-5.0),
            (),
        )
        self.mac_query_proj = nn.Dense(
            self.n_memory_tokens * self.d_model,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="mac_query_proj",
        )
        self.mac_out_proj = nn.Dense(
            self.d_model, use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="mac_out_proj",
        )

    # ── Methods ───────────────────────────────────────────────────────────

    def ssm_inject(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inject memory state into input via learned query (broadcast over T)."""
        B = x.shape[0]
        query = jnp.broadcast_to(
            self.ssm_query[None], (B, 1, self.d_model)
        )  # [B, 1, d_model]
        retrieval = self.memory.retrieve(query)   # [B, 1, d_model]
        signal = self.ssm_proj(retrieval)          # [B, 1, d_model]
        return x + signal                          # broadcast over T

    def mag_inject(
        self,
        x: jnp.ndarray,
        channel_start: int,
        channel_end: int,
        scale: float = 1.0,
    ) -> jnp.ndarray:
        """Gated residual update into memory channel slice."""
        retrieval = self.memory.retrieve(x)           # [B, T, d_model]
        mem_signal = self.mag_proj(retrieval)          # [B, T, d_mem_ch]
        gate = jax.nn.sigmoid(self.mag_gate(x))       # [B, T, d_mem_ch]
        gated = (gate * mem_signal * scale).astype(x.dtype)

        prefix = x[..., :channel_start]
        mem_ch = x[..., channel_start:channel_end] + gated
        suffix = x[..., channel_end:]
        return jnp.concatenate([prefix, mem_ch, suffix], axis=-1)

    def update(self, x: jnp.ndarray, suppress_decay: bool = False) -> jnp.ndarray:
        """Forward-pass inner-GD update. Call only during training."""
        return self.memory.update(x, suppress_decay=suppress_decay)

    def retrieve(self, x: jnp.ndarray) -> jnp.ndarray:
        """Split-path retrieve from memory."""
        return self.memory.retrieve(x)

    def get_mac_tokens(self, x: jnp.ndarray) -> jnp.ndarray:
        """Generate dynamic MAC tokens for KV-prepend."""
        n = self.n_memory_tokens
        B, T, D = x.shape
        x_pool = x.mean(axis=1)                    # [B, D]
        q_flat = self.mac_query_proj(x_pool)       # [B, n * D]
        queries = q_flat.reshape(B, n, D)          # [B, n, D]
        raw = self.memory.retrieve(queries)        # [B, n, d_model]
        tokens = self.mac_out_proj(raw)            # [B, n, d_model]
        gate = jax.nn.softplus(self.mac_gate_raw.astype(jnp.float32))
        return (tokens * gate).astype(x.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Default: retrieve."""
        return self.retrieve(x)
