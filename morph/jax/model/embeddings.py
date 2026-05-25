"""MORPH embedding modules — JAX/Flax port.

Exact port of morph/model/embeddings.py to Flax linen.

Three embedding components, always on:
  - LorentzEmbedding     — Lorentz hyperboloid (space → hyperboloid → tangent)
  - HybridEmbedding      — Concatenated euclidean + Lorentz, split by lorentz_fraction
  - BigramEmbedding      — Hash-based 2-gram position-invariant embedding
  - MORPHEmbedding       — Combines HybridEmbedding + BigramEmbedding, weight-tied LM head

No runtime if-statements. All features always active.
bf16 compatible. XLA-friendly (no in-place ops).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import flax.linen as nn


_EPS = 1e-6

# ── Lorentz helpers ───────────────────────────────────────────────────────────


def project_to_hyperboloid(space: jnp.ndarray) -> jnp.ndarray:
    """Map a Euclidean vector to the Lorentz hyperboloid x₀² - ||xs||² = 1.

    Args:
        space: [..., d] spatial (non-time) components.

    Returns:
        [..., d+1] full Lorentz vector [x₀, xs].
    """
    sq_norm = (space * space).sum(axis=-1, keepdims=True)
    x0 = jnp.sqrt(jnp.maximum(1.0 + sq_norm, _EPS))
    return jnp.concatenate([x0, space], axis=-1)


def log_map_origin(x: jnp.ndarray) -> jnp.ndarray:
    """Logarithmic map at the origin of the Lorentz model → tangent space.

    Drops the time component x₀ and returns a d-dimensional tangent vector.

    Near the origin denom→0; coefficient → 1 (L'Hôpital).

    Args:
        x: [..., d+1] Lorentz vector (x₀ is first component).

    Returns:
        [..., d] tangent vector at the origin.
    """
    x0 = x[..., :1]                                              # time component
    xs = x[..., 1:]                                              # spatial components
    alpha = jnp.arccosh(jnp.maximum(x0, 1.0 + _EPS))            # geodesic distance
    denom = jnp.sqrt(jnp.maximum(x0 * x0 - 1.0, _EPS))
    coeff = jnp.where(denom < 1e-4, jnp.ones_like(denom), alpha / denom)
    return coeff * xs


# ── LorentzEmbedding ──────────────────────────────────────────────────────────


class LorentzEmbedding(nn.Module):
    """Embedding on the Lorentz hyperboloid, projected to the tangent space.

    Pipeline: token_id → space_embed (d-dim) → project_to_hyperboloid (d+1-dim)
              → log_map_origin (d-dim tangent vector).

    attend() maps a hidden state back to logits via the weight-tied tangent
    weight matrix.

    Attributes
    ----------
    num_embeddings : int   Vocabulary size.
    features       : int   Spatial dimension d. Output dim = d (not d+1).
    """

    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Return tangent-space embedding, shape [..., features]."""
        space_embed = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            embedding_init=nn.initializers.normal(stddev=0.005),
            name="space_embed",
        )
        space = space_embed(input_ids)
        return log_map_origin(project_to_hyperboloid(space))

    def attend(self, x: jnp.ndarray) -> jnp.ndarray:
        """Weight-tied LM head: hidden state → vocab logits.

        Args:
            x: [..., features] hidden state.

        Returns:
            [..., num_embeddings] logits.
        """
        space_w = self.variables["params"]["space_embed"]["embedding"]  # [V, features]
        tangent_w = log_map_origin(project_to_hyperboloid(space_w))     # [V, features]
        return x @ tangent_w.T


# ── HybridEmbedding ───────────────────────────────────────────────────────────


class HybridEmbedding(nn.Module):
    """Concatenated Euclidean + Lorentz embedding.

    Output: [euc_embed(ids) || lor_embed(ids)], shape [..., d_model].

    Attributes
    ----------
    vocab_size        : int
    d_model           : int   Total output dimension.
    lorentz_fraction  : float Fraction of d_model for the Lorentz channel.
                              lorentz_dim = int(d_model * lorentz_fraction).
                              euclidean_dim = d_model - lorentz_dim.
    """

    vocab_size: int
    d_model: int
    lorentz_fraction: float

    def setup(self):
        lor_dim = int(self.d_model * self.lorentz_fraction)
        euc_dim = self.d_model - lor_dim
        assert euc_dim > 0, (
            f"lorentz_fraction={self.lorentz_fraction} leaves no room for euclidean dims"
        )
        assert lor_dim > 0, (
            f"lorentz_fraction={self.lorentz_fraction} leaves no room for lorentz dims"
        )
        self._lorentz_dim = lor_dim
        self._euclidean_dim = euc_dim

        self.euc_embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=euc_dim,
            embedding_init=nn.initializers.normal(stddev=1.0 / math.sqrt(euc_dim)),
            name="euc_embed",
        )
        self.lor_embed = LorentzEmbedding(
            num_embeddings=self.vocab_size,
            features=lor_dim,
            name="lor_embed",
        )

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Return [euclidean || lorentz] embedding, shape [..., d_model]."""
        euc = self.euc_embed(input_ids)
        lor = self.lor_embed(input_ids)
        return jnp.concatenate([euc, lor], axis=-1)

    def attend(self, x: jnp.ndarray) -> jnp.ndarray:
        """Weight-tied LM head logits, shape [..., vocab_size].

        Splits x into euclidean and lorentz slices, computes logits from each
        weight matrix, and sums them for the final score.
        """
        x_euc = x[..., :self._euclidean_dim]
        x_lor = x[..., self._euclidean_dim:]

        euc_w = self.variables["params"]["euc_embed"]["embedding"]  # [V, euc_dim]

        lor_space_w = self.variables["params"]["lor_embed"]["space_embed"]["embedding"]
        lor_hyp = project_to_hyperboloid(lor_space_w)
        lor_w = log_map_origin(lor_hyp)                             # [V, lor_dim]

        # Full vocab weight matrix: [V, d_model]
        w_full = jnp.concatenate([euc_w, lor_w], axis=-1)
        return x @ w_full.T


# ── BigramEmbedding ───────────────────────────────────────────────────────────

# Hash constants matching the PyTorch implementation exactly.
_BIGRAM_HASH_A: int = 279470273
_BIGRAM_HASH_B: int = 4294967291


class BigramEmbedding(nn.Module):
    """Hash-based 2-gram embedding with per-layer learned lambda scaling.

    Each token's bigram key is: (A * token_id XOR B * prev_token_id) % hash_vocab.
    Position-invariant, captures local n-gram statistics without an explicit
    bigram table (hash collision rate ~1/hash_vocab).

    Per-layer lambdas start at 0 (no bigram signal at init) and are learned.
    Use compute() once per forward pass, then inject() at each layer.

    Attributes
    ----------
    vocab_size  : int   Model vocabulary size (unused in hashing, kept for API parity).
    d_model     : int   Embedding dimension.
    n_layers    : int   Total number of layers (prelude + core + coda).
    hash_vocab  : int   Hash table size.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    hash_vocab: int = 49152

    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=self.hash_vocab,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="embed",
        )
        # Per-layer scalar lambdas — init 0 so bigram has no effect at start.
        self.lambdas = self.param(
            "lambdas",
            nn.initializers.zeros,
            (self.n_layers,),
        )

    def compute(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Compute bigram embedding for every token position.

        Args:
            input_ids: [B, S] integer token ids.

        Returns:
            [B, S, d_model] bigram embeddings.
        """
        # Shift right by one, pad with 0 for the first position
        prev = jnp.pad(input_ids[:, :-1], ((0, 0), (1, 0)), constant_values=0)
        # Hash constants may exceed int32. Use numpy for compile-time constants,
        # then bring into JAX as int64 to avoid overflow.
        # JAX x64 may not be enabled; we use jnp.array with explicit dtype.
        import numpy as _np
        a = _np.int64(_BIGRAM_HASH_A)
        b = _np.int64(_BIGRAM_HASH_B)
        hv = _np.int64(self.hash_vocab)
        ids64 = input_ids.astype(jnp.int64)
        prev64 = prev.astype(jnp.int64)
        # Perform arithmetic in int64 numpy scalars broadcast over the arrays
        hash_ids = ((a * ids64) ^ (b * prev64)) % hv
        hash_ids = hash_ids.astype(jnp.int32)
        return self.embed(hash_ids)

    def inject(
        self,
        x: jnp.ndarray,
        bigram_emb: jnp.ndarray,
        layer_idx: int,
    ) -> jnp.ndarray:
        """Add scaled bigram signal to the residual stream.

        Args:
            x:          [B, S, d_model] residual stream.
            bigram_emb: [B, S, d_model] output of compute().
            layer_idx:  which lambda scalar to use.

        Returns:
            [B, S, d_model] updated residual stream (XLA-safe, no in-place).
        """
        lam = self.lambdas[layer_idx].astype(x.dtype)
        return x + lam * bigram_emb.astype(x.dtype)


# ── MORPHEmbedding ────────────────────────────────────────────────────────────


class MORPHEmbedding(nn.Module):
    """Complete MORPH embedding module: HybridEmbedding + BigramEmbedding.

    All features always on:
      - HybridEmbedding provides the input token representation
        (Euclidean || Lorentz channels, always both active).
      - BigramEmbedding provides per-layer bigram injection signals.
      - attend() is the weight-tied LM head.

    Attributes
    ----------
    vocab_size         : int
    d_model            : int   Model hidden dimension.
    lorentz_fraction   : float Fraction of d_model for the Lorentz channel.
    bigram_hash_vocab  : int   Hash table size for bigram embeddings.
    n_layers           : int   Total layers (for per-layer bigram lambdas).
    """

    vocab_size: int
    d_model: int
    lorentz_fraction: float
    bigram_hash_vocab: int
    n_layers: int

    def setup(self):
        self.hybrid = HybridEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            lorentz_fraction=self.lorentz_fraction,
            name="hybrid",
        )
        self.bigram = BigramEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            hash_vocab=self.bigram_hash_vocab,
            name="bigram",
        )

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Return hybrid (euclidean+lorentz) embedding, shape [B, S, d_model].

        Call this once at the start of the forward pass to get the input
        representation. Bigram injections are applied per-layer via get_bigram()
        + bigram.inject().
        """
        return self.hybrid(input_ids)

    def get_bigram(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Compute bigram embeddings to be injected at each layer.

        Args:
            input_ids: [B, S] integer token ids.

        Returns:
            [B, S, d_model] bigram embeddings. Pass to bigram.inject() at each
            layer with the appropriate layer_idx.
        """
        return self.bigram.compute(input_ids)

    def attend(self, x: jnp.ndarray) -> jnp.ndarray:
        """Weight-tied LM head: hidden state → vocab logits.

        Args:
            x: [B, S, d_model] final hidden state (after norm).

        Returns:
            [B, S, vocab_size] logits.
        """
        return self.hybrid.attend(x)
