"""STP + LeJEPA split_nsm z-latent prediction modules (JAX/Flax).

Port of the STP and z-latent (split_nsm) systems from:
  TPU/subq-attention/looped_model_mhc.py  (PyTorch)
  titans_core/memory/sigreg.py            (SIGReg, PyTorch)

Modules
-------
SIGReg
    Epps-Pulley characteristic function anti-collapse regularizer.
    Enforces isotropic Gaussian distribution on embedding vectors.
    No learned parameters. Zero-shot prevention of representational collapse.

STPLoss
    Semantic Tube Prediction (Huang, LeCun, Balestriero 2026).
    Full-sequence vectorized geodesic smoothness regularizer.
    Modes:
      "geodesic"    — multi-scale strides {1, 2, 4, ..., tau}, tau=64.
      "consecutive" — stride-1 all positions.
    Loss: 1 - cos(v_fwd, v_bwd) averaged over positions and scales.
    Zero parameters.

ZLatentHeads
    LeJEPA split_nsm dual-head prediction system.
    Backbone z-head: predicts mean of next segment's z_coda.
    Memory z-head:   predicts next segment's prelude state.
    Both computed in the outer segmented loop (delayed cross-segment targets).
    SIGReg prevents collapse without detaching targets.

SplitNSMOuterLoop
    Orchestrates the delayed z-loss computation for split_nsm mode.
    Accepts per-segment outputs (z_coda, x_prelude_raw, z_prelude, z_memory_raw)
    accumulated across segments and computes the final cross-segment loss.
    Stateless function-style API — no learned parameters.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


# ── SIGReg ─────────────────────────────────────────────────────────────────────


class SIGReg(nn.Module):
    """Epps-Pulley characteristic function test for anti-collapse regularization.

    Computes a statistic measuring deviation of the input distribution from
    an isotropic Gaussian. Returns a scalar loss — add to total loss with
    weight lambda (default 0.02).

    No learnable parameters. Quadrature points and projection directions are
    regenerated each call for stochasticity of the random projections.

    Attributes
    ----------
    knots : int
        Number of quadrature points on [0, 3].  Default 17.
    n_slices : int
        Number of random projection slices.  Default 256.
    """

    knots: int = 17
    n_slices: int = 256

    @nn.compact
    def __call__(self, proj: jnp.ndarray, rng: jax.Array | None = None) -> jnp.ndarray:
        """Compute SIGReg statistic.

        Args:
            proj: Embedding tensor [..., N, D] where N is batch/token dim
                  and D is embedding dim. Operates on last two dims.
            rng:  Optional PRNGKey for random projections. If None, uses a
                  fresh key derived from make_rng("sigreg") — must be in a
                  scope with that collection available, else pass rng explicitly.

        Returns:
            Scalar loss measuring deviation from isotropic Gaussian.
        """
        # Quadrature points on [0, 3]
        t = jnp.linspace(0.0, 3.0, self.knots, dtype=jnp.float32)  # [knots]
        dt = 3.0 / (self.knots - 1)
        weights = jnp.full((self.knots,), 2.0 * dt, dtype=jnp.float32)
        weights = weights.at[0].set(dt).at[-1].set(dt)
        phi = jnp.exp(-t ** 2 / 2.0)  # Gaussian CF
        weighted = weights * phi       # [knots]

        # Random projection directions
        if rng is None:
            rng = self.make_rng("sigreg")
        A = jax.random.normal(rng, (proj.shape[-1], self.n_slices), dtype=jnp.float32)
        A = A / (jnp.linalg.norm(A, axis=0, keepdims=True) + 1e-8)  # [D, n_slices]

        # Project embeddings: [..., N, n_slices]
        proj_f32 = proj.astype(jnp.float32)
        projected = proj_f32 @ A  # [..., N, n_slices]

        # Characteristic function test at quadrature points
        # x_t: [..., N, n_slices, knots]
        x_t = projected[..., None] * t  # broadcast t over last dim

        # Empirical CF vs Gaussian CF (mean over N dim = -3 after unsqueeze)
        cos_mean = jnp.cos(x_t).mean(axis=-3)  # [..., n_slices, knots]
        sin_mean = jnp.sin(x_t).mean(axis=-3)
        err = (cos_mean - phi) ** 2 + sin_mean ** 2  # [..., n_slices, knots]

        # Weighted integral, scaled by sample size
        statistic = (err @ weighted) * proj.shape[-2]  # [..., n_slices]
        return statistic.mean()


# ── STPLoss ────────────────────────────────────────────────────────────────────


class STPLoss(nn.Module):
    """Semantic Tube Prediction (STP) loss — zero parameters.

    Enforces geodesic linearity in the representation manifold:
    for any triplet (a, b, c) along a trajectory, the forward difference
    b→c should align with the backward difference a→b in cosine similarity.

    Modes
    -----
    "geodesic"
        Multi-scale: stride k triplets for k in {1, 2, 4, ..., tau}, tau ≤ stp_tau.
        Tests representational linearity at multiple scales simultaneously.
        Loss = mean over all scales and positions of (1 - cosine_similarity).

    "consecutive"
        Stride-1 only: consecutive triplets (t-2, t-1, t).
        Loss = mean over all positions of (1 - cosine_similarity).

    Attributes
    ----------
    mode    : str   "geodesic" | "consecutive"
    tau     : int   Maximum stride (geodesic mode). Power-of-2 halting.
    lam     : float Loss weight applied before adding to total loss.
    """

    mode: str = "geodesic"
    tau: int = 64
    lam: float = 0.02

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        """Compute STP loss.

        Args:
            h: [B, T, D] hidden states (float32 recommended; cast internally).

        Returns:
            Scalar loss (unweighted by lam — caller multiplies by self.lam).
        """
        h = h.astype(jnp.float32)  # STP in fp32 for numerical stability
        T = h.shape[1]

        if self.mode == "consecutive":
            return self._consecutive_loss(h, T)
        elif self.mode == "geodesic":
            return self._geodesic_loss(h, T)
        else:
            raise ValueError(f"Unknown STP mode: {self.mode!r}. Use 'geodesic' or 'consecutive'.")

    def _cosine_loss(self, v_fwd: jnp.ndarray, v_bwd: jnp.ndarray) -> jnp.ndarray:
        """1 - cos_sim(v_fwd, v_bwd), averaged over batch and position."""
        eps = 1e-8
        norm_fwd = jnp.linalg.norm(v_fwd, axis=-1, keepdims=True)
        norm_bwd = jnp.linalg.norm(v_bwd, axis=-1, keepdims=True)
        dot = (v_fwd * v_bwd).sum(axis=-1)
        cos_sim = dot / (norm_fwd.squeeze(-1) * norm_bwd.squeeze(-1) + eps)
        return (1.0 - cos_sim).mean()

    def _consecutive_loss(self, h: jnp.ndarray, T: int) -> jnp.ndarray:
        if T < 3:
            return jnp.zeros((), dtype=jnp.float32)
        v_fwd = h[:, 2:] - h[:, 1:-1]   # [B, T-2, D]
        v_bwd = h[:, 1:-1] - h[:, :-2]   # [B, T-2, D]
        return self._cosine_loss(v_fwd, v_bwd)

    def _geodesic_loss(self, h: jnp.ndarray, T: int) -> jnp.ndarray:
        if T < 3:
            return jnp.zeros((), dtype=jnp.float32)

        tau = min(self.tau, T // 2)
        losses = []
        k = 1
        while k <= tau and T > 2 * k:
            v_fwd = h[:, 2*k:] - h[:, k:-k]       # [B, T-2k, D]
            v_bwd = h[:, k:-k] - h[:, :-2*k]      # [B, T-2k, D]
            losses.append(self._cosine_loss(v_fwd, v_bwd))
            k *= 2

        if not losses:
            return jnp.zeros((), dtype=jnp.float32)

        return jnp.stack(losses).mean()


# ── Z-latent heads ─────────────────────────────────────────────────────────────


class ZLatentHeads(nn.Module):
    """LeJEPA split_nsm dual-head prediction system.

    Manages the two prediction heads used in split_nsm mode:

      z_projector    : projects coda hidden states → latent space (d_z dims).
                       Used as the self-supervised target (no EMA teacher).
      backbone_head  : linear d_model → d_z applied to x_prelude_raw.
                       Predicts mean of NEXT segment's z_coda.
      memory_head    : linear d_model → d_z applied to memory retrieval.
                       Predicts NEXT segment's prelude state (in z-latent space).

    SIGReg prevents collapse on the targets — no detach needed per LeJEPA.

    Usage: call setup() implicitly by calling the module (all three Dense layers
    are created in setup so they always appear in params together). Then call
    individual methods (project_coda, backbone_predict, memory_predict) via
    module.apply() or directly on a bound module.

    Attributes
    ----------
    d_model        : int
    d_z            : int   Latent dimension for prediction heads.
    sigreg_knots   : int
    sigreg_n_slices: int
    """

    d_model: int
    d_z: int = 256
    sigreg_knots: int = 17
    sigreg_n_slices: int = 256

    def setup(self):
        # All three Dense layers declared here so params are always initialized together.
        init = nn.initializers.lecun_uniform()
        self.z_proj_linear  = nn.Dense(self.d_z, use_bias=True, kernel_init=init)
        self.z_proj_norm    = nn.LayerNorm()
        self.z_backbone_head = nn.Dense(self.d_z, use_bias=False, kernel_init=init)
        self.z_memory_head  = nn.Dense(self.d_z, use_bias=False, kernel_init=init)
        self._sigreg = SIGReg(knots=self.sigreg_knots, n_slices=self.sigreg_n_slices)

    def __call__(
        self,
        x_coda_normed: jnp.ndarray,
        x_prelude_raw: jnp.ndarray,
        mem_retrieval: jnp.ndarray,
        rng_sigreg: jax.Array,
    ) -> dict[str, jnp.ndarray]:
        """Compute all three predictions in one call (preferred for init).

        Args:
            x_coda_normed: [B, T, d_model] coda output after final_norm.
            x_prelude_raw: [B, T, d_model] raw prelude output.
            mem_retrieval: [B, T, d_model] memory retrieval output.
            rng_sigreg:    PRNGKey for SIGReg random projections.

        Returns:
            dict with keys:
              "z_coda"     : [B, T, d_z]
              "bb_pred"    : [B, T, d_z]
              "z_memory"   : [B, T, d_z]
              "sigreg_loss": scalar
        """
        z_coda   = self.z_proj_norm(self.z_proj_linear(x_coda_normed))
        bb_pred  = self.z_backbone_head(x_prelude_raw)
        z_memory = self.z_memory_head(mem_retrieval)
        sig_loss = self._sigreg(z_coda, rng=rng_sigreg)
        return {
            "z_coda":      z_coda,
            "bb_pred":     bb_pred,
            "z_memory":    z_memory,
            "sigreg_loss": sig_loss,
        }

    def project_coda(self, x_coda_normed: jnp.ndarray) -> jnp.ndarray:
        """Project coda output → latent space. Called per segment.

        Args:
            x_coda_normed: [B, T, d_model] (after final_norm).

        Returns:
            z_coda: [B, T, d_z]
        """
        return self.z_proj_norm(self.z_proj_linear(x_coda_normed))

    def project_prelude(self, x_prelude_normed: jnp.ndarray) -> jnp.ndarray:
        """Project prelude output → latent space (same projector as coda)."""
        return self.z_proj_norm(self.z_proj_linear(x_prelude_normed))

    def backbone_predict(self, x_prelude_raw: jnp.ndarray) -> jnp.ndarray:
        """Backbone prediction: prelude hidden → next segment z_coda.

        Args:
            x_prelude_raw: [B, T, d_model] raw prelude output (pre-projection).

        Returns:
            bb_pred: [B, T, d_z]
        """
        return self.z_backbone_head(x_prelude_raw)

    def memory_predict(self, mem_retrieval: jnp.ndarray) -> jnp.ndarray:
        """Memory prediction: memory retrieval → next segment prelude state.

        Args:
            mem_retrieval: [B, T, d_model] memory retrieval output.

        Returns:
            z_memory: [B, T, d_z]
        """
        return self.z_memory_head(mem_retrieval)

    def sigreg_loss(self, z: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        """Compute SIGReg anti-collapse loss on a latent tensor."""
        return self._sigreg(z, rng=rng)


# ── Split NSM outer loop loss ──────────────────────────────────────────────────


def split_nsm_outer_loss(
    seg_z_codas: Sequence[jnp.ndarray],
    seg_bb_preds: Sequence[jnp.ndarray],
    seg_z_preludes: Sequence[jnp.ndarray],
    seg_z_memories: Sequence[jnp.ndarray],
    rng: jax.Array,
    sigreg_knots: int = 17,
    sigreg_n_slices: int = 256,
    sigreg_weight: float = 0.02,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute the delayed cross-segment z-latent losses for split_nsm mode.

    Pure function — no module references. All inputs are pre-projected JAX arrays.
    Called AFTER all segments have been processed (outer segmented loop).

    Backbone and memory projections happen per-segment during the forward pass
    (via ZLatentHeads methods inside apply scope). This function only computes
    the cross-segment pairing loss and SIGReg regularization.

    Both loss components use smooth-L1 (Huber) on segment-mean representations
    for stable cross-segment signal under variable sequence lengths.

    Split NSM assignment:
      Backbone loss: bb_preds[i] (projected prelude) → predict mean(z_coda[i+1])
      Memory loss:   z_memories[i]                  → predict z_prelude[i+1]

    SIGReg is applied to seg_z_codas[1] (the first cross-segment target).
    No target detach — SIGReg is the sole anti-collapse mechanism (LeJEPA).

    Args:
        seg_z_codas   : list of [B, T, d_z] z_coda per segment (from z_projector).
        seg_bb_preds  : list of [B, T, d_z] backbone predictions per segment
                        (from z_backbone_head(x_prelude_raw) — projected per segment).
        seg_z_preludes: list of [B, T, d_z] projected prelude per segment
                        (from z_projector(final_norm(x_prelude))).
        seg_z_memories: list of [B, T, d_z] memory head predictions per segment
                        (from z_memory_head(mem_retrieval)).
        rng           : JAX PRNGKey for SIGReg random projections.
        sigreg_knots  : SIGReg quadrature points (default 17).
        sigreg_n_slices: SIGReg random projections (default 256).
        sigreg_weight : SIGReg loss coefficient (default 0.02).

    Returns:
        (z_fwd_loss, metrics)
        z_fwd_loss : scalar JAX float32 array.
        metrics    : dict with "bb_loss", "mem_loss", "sigreg_loss" for logging.
    """
    n_pairs = min(
        len(seg_z_memories),
        len(seg_z_codas) - 1,
    )

    zero = jnp.zeros((), dtype=jnp.float32)
    if n_pairs < 1:
        return zero, {"bb_loss": zero, "mem_loss": zero, "sigreg_loss": zero}

    bb_losses = []
    mem_losses = []

    for i in range(n_pairs):
        # ── Backbone: bb_preds[i] → predict mean(z_coda[i+1]) ──────────────
        if i < len(seg_bb_preds):
            bb_pred   = seg_bb_preds[i]                                    # [B, T, d_z]
            bb_target = seg_z_codas[i + 1].mean(axis=1, keepdims=True)    # [B, 1, d_z]
            bb_pred_mean = bb_pred.mean(axis=1, keepdims=True)             # [B, 1, d_z]
            bb_losses.append(_smooth_l1(bb_pred_mean, bb_target))

        # ── Memory: z_memories[i] → predict z_prelude[i+1] ─────────────────
        if i + 1 < len(seg_z_preludes):
            mem_mean       = seg_z_memories[i].mean(axis=1, keepdims=True)       # [B, 1, d_z]
            prelude_target = seg_z_preludes[i + 1].mean(axis=1, keepdims=True)   # [B, 1, d_z]
            mem_losses.append(_smooth_l1(mem_mean, prelude_target))

    # SIGReg on first cross-segment z_coda target (anti-collapse, applied once).
    rng, rng_sr = jax.random.split(rng)
    sigreg = SIGReg(knots=sigreg_knots, n_slices=sigreg_n_slices)
    # SIGReg has no params; use empty variables.
    sigreg_val = sigreg.apply({}, seg_z_codas[1].astype(jnp.float32), rng=rng_sr)

    bb_loss  = jnp.stack(bb_losses).mean()  if bb_losses  else zero
    mem_loss = jnp.stack(mem_losses).mean() if mem_losses else zero
    z_fwd_loss = bb_loss + mem_loss + sigreg_weight * sigreg_val

    metrics = {
        "bb_loss":     bb_loss,
        "mem_loss":    mem_loss,
        "sigreg_loss": sigreg_val,
    }
    return z_fwd_loss, metrics


# ── Utility ────────────────────────────────────────────────────────────────────


def _smooth_l1(pred: jnp.ndarray, target: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """Huber / smooth-L1 loss, scalar output.

    Matches PyTorch F.smooth_l1_loss(reduction='mean', beta=1.0).
    """
    diff = pred - target
    abs_diff = jnp.abs(diff)
    loss = jnp.where(abs_diff < beta, 0.5 * diff ** 2 / beta, abs_diff - 0.5 * beta)
    return loss.mean()
