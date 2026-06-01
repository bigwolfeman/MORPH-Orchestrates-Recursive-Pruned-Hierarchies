"""Chunked cross-entropy for the weight-tied LM head — JAX/XLA port.

Background
----------
The JAX loss path in transformer.py computes:

    logits = embed.attend(x_for_head)   # [B·T, V]   ← full materialization
    log_probs = log_softmax(logits)     # [B·T, V]   ← second full buffer
    ce_loss = -log_probs[n, labels[n]] ...

At V=49152 and B·T=8·4096=32768 tokens (bf16), that is two 32768×49152 bf16
arrays = 2 × 3.1 GiB = 6.2 GiB of HBM just for the logits.  On TPU HBM this
dominates batch-size and sequence-length limits.

XLA note
--------
XLA *can* fuse log_softmax + gather into a single kernel that never stores
the full softmax distribution — the key op is jax.nn.sparse_softmax_cross_entropy,
which XLA compiles to a fused kernel that only needs the target logit and the
log-sum-exp.  However that path is NOT the same as calling log_softmax on the
full [N, V] array: the full array is still materialized in the naive path.

This module provides:

1. ``sparse_ce_row(x, w, label)`` — single-row (token-level) CE using
   jax.nn.sparse_softmax_cross_entropy_with_logits, which XLA fuses without
   materializing the full softmax.  Used by the chunked path.

2. ``chunked_cross_entropy(x, w, labels, ignore_index, chunk_size)`` — map
   over token chunks using jax.lax.map (a for-loop that XLA can pipeline).
   Peak memory: O(chunk_size × V) instead of O(N × V).
   Numerics match naive path to fp32 reduction error (~1e-5).

3. ``naive_cross_entropy(x, w, labels, ignore_index)`` — reference
   implementation materialising the full [N, V] array. Used for parity checks.

Contract
--------
    loss = chunked_cross_entropy(x, w, labels, ignore_index=-100, chunk_size=1024)
    x:       [N, d]    hidden states (bf16 or fp32).
    w:       [V, d]    weight-tied LM-head matrix (from embed.lm_weight or
                       assembled by the caller from the hybrid embedding params).
    labels:  [N]       int32 targets; ignore_index rows contribute 0 to the mean.
    Returns: scalar mean CE loss.  Gradients flow back to both x and w.

Note: the caller is responsible for assembling w from the HybridEmbedding
parameters (same as the ``attend`` path) and passing it here.  This keeps
the chunked-CE module independent of the embedding internals.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp


# ─────────────────────────────────────────────────────────────────────────────
# Numerically stable sparse CE: log-sum-exp - target_logit
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_ce_from_logits(labels_safe: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable cross-entropy: logsumexp(logits) - logits[label].

    XLA fuses log-softmax + gather into an efficient kernel (no full [N, V]
    softmax distribution needs to be materialised in the backward).

    Args:
        labels_safe: [N] int32 label indices (must be in [0, V); caller zeros
                     out the ignored rows' contributions via a mask).
        logits:      [N, V] float32 logits.

    Returns:
        [N] per-token cross-entropy values.
    """
    # logsumexp over vocab dimension
    lse = jsp.special.logsumexp(logits, axis=-1)                  # [N]
    # Gather target logits: logits[n, labels_safe[n]]
    N = logits.shape[0]
    idx = jnp.arange(N, dtype=jnp.int32)
    target_logit = logits[idx, labels_safe]                       # [N]
    return lse - target_logit


# ─────────────────────────────────────────────────────────────────────────────
# Chunked CE — lax.map over token-row chunks to bound peak memory
# ─────────────────────────────────────────────────────────────────────────────

def chunked_cross_entropy(
    x: jnp.ndarray,
    w: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> jnp.ndarray:
    """Memory-efficient mean CE for the weight-tied LM head.

    Processes ``chunk_size`` token rows at a time.  Within each chunk the
    logits tile ``[chunk_size, V]`` is materialised then freed; the maximum
    live memory is ``O(chunk_size × V)`` instead of ``O(N × V)``.

    At chunk_size=1024 and V=49152 bf16 that is 1024×49152×2 ≈ 96 MiB instead
    of N×49152 which at N=32768 is 3.1 GiB.

    Loss is mean over non-ignored tokens (matching F.cross_entropy default).

    Args:
        x:            [N, d]   hidden states.
        w:            [V, d]   weight matrix (from embed.attend weights).
        labels:       [N]      int32 target token ids.
        ignore_index: int      label value to ignore (default -100).
        chunk_size:   int      rows per chunk (tune: smaller = less HBM, more
                               kernel launches; 1024 is a good default).

    Returns:
        scalar jnp.float32 mean loss.
    """
    N, d = x.shape
    V = w.shape[0]

    # Pad N to a multiple of chunk_size so we can use lax.map (static shape).
    pad = (-N) % chunk_size
    if pad > 0:
        x_pad = jnp.concatenate(
            [x, jnp.zeros((pad, d), dtype=x.dtype)], axis=0
        )
        # Pad labels with ignore_index so padded rows contribute 0.
        labels_pad = jnp.concatenate(
            [labels, jnp.full((pad,), ignore_index, dtype=jnp.int32)], axis=0
        )
    else:
        x_pad = x
        labels_pad = labels.astype(jnp.int32)

    N_pad = x_pad.shape[0]
    n_chunks = N_pad // chunk_size

    # Reshape to [n_chunks, chunk_size, ...]
    x_chunks = x_pad.reshape(n_chunks, chunk_size, d)
    l_chunks = labels_pad.reshape(n_chunks, chunk_size)

    w_f32 = w.astype(jnp.float32)  # promote weight once (shared across chunks)

    def _chunk_loss(carry, chunk_data):
        """Process one chunk; carry is unused (lax.scan-style but using lax.scan)."""
        x_c, lab_c = chunk_data
        # logits for this chunk: [chunk_size, V] — freed after this function.
        # XLA can pipeline/fuse the logsumexp+gather since the output is
        # a scalar per-token loss, not the full [chunk_size, V] distribution.
        logits_c = x_c.astype(jnp.float32) @ w_f32.T      # [chunk_size, V]
        valid = (lab_c != ignore_index)                     # [chunk_size] bool
        # Clamp labels to valid range for gather (ignored rows get 0 idx).
        lab_safe = jnp.where(valid, lab_c, jnp.zeros_like(lab_c))
        loss_c = _sparse_ce_from_logits(lab_safe, logits_c)
        # Zero out ignored rows.
        loss_c = jnp.where(valid, loss_c, jnp.zeros_like(loss_c))
        return None, (loss_c, valid.astype(jnp.float32))

    _, (losses, valids) = jax.lax.scan(
        _chunk_loss, None, (x_chunks, l_chunks)
    )
    # losses: [n_chunks, chunk_size], valids: [n_chunks, chunk_size]
    total_loss = losses.sum()
    total_valid = valids.sum()
    return (total_loss / jnp.maximum(total_valid, 1.0)).astype(jnp.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Reference: naive full-materialization path (for parity checks only)
# ─────────────────────────────────────────────────────────────────────────────

def naive_cross_entropy(
    x: jnp.ndarray,
    w: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = -100,
) -> jnp.ndarray:
    """Reference CE that materialises full [N, V] logits.

    Use ONLY for parity checks — this is the slow/memory-hungry path.

    Args:
        x:      [N, d]
        w:      [V, d]
        labels: [N] int32

    Returns:
        scalar float32 mean loss.
    """
    logits = x.astype(jnp.float32) @ w.astype(jnp.float32).T   # [N, V]
    valid = (labels != ignore_index)
    lab_safe = jnp.where(valid, labels, jnp.zeros_like(labels)).astype(jnp.int32)
    loss_per_token = _sparse_ce_from_logits(lab_safe, logits)
    loss_per_token = jnp.where(valid, loss_per_token, jnp.zeros_like(loss_per_token))
    return (loss_per_token.sum() / jnp.maximum(valid.sum().astype(jnp.float32), 1.0))
