"""MORPH embedding modules.

Three embedding components, always on:
  - LorentzEmbedding     — Lorentz hyperboloid (space → hyperboloid → tangent)
  - HybridEmbedding      — Concatenated euclidean + Lorentz, split by lorentz_fraction
  - BigramEmbedding      — Hash-based 2-gram position-invariant embedding
  - MORPHEmbedding       — Combines HybridEmbedding + BigramEmbedding, weight-tied LM head

No runtime if-statements. All features always active.
bf16 compatible. torch.compile friendly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── Lorentz helpers ───────────────────────────────────────────────────────────

_EPS = 1e-6


def _project_to_hyperboloid(space: Tensor) -> Tensor:
    """Map a Euclidean vector to the Lorentz hyperboloid x₀² - ||xs||² = 1.

    Args:
        space: [..., d] spatial (non-time) components.

    Returns:
        [..., d+1] full Lorentz vector [x₀, xs].
    """
    sq_norm = (space * space).sum(dim=-1, keepdim=True)
    x0 = torch.sqrt(torch.clamp(1.0 + sq_norm, min=_EPS))
    return torch.cat([x0, space], dim=-1)


def _log_map_origin(x: Tensor) -> Tensor:
    """Logarithmic map at the origin of the Lorentz model → tangent space.

    Drops the time component x₀ and returns a d-dimensional tangent vector.

    Args:
        x: [..., d+1] Lorentz vector (x₀ is first component).

    Returns:
        [..., d] tangent vector at the origin.
    """
    x0 = x[..., :1]                                            # time component
    xs = x[..., 1:]                                            # spatial components
    alpha = torch.acosh(torch.clamp(x0, min=1.0 + _EPS))      # geodesic distance
    denom = torch.sqrt(torch.clamp(x0 * x0 - 1.0, min=_EPS))
    # Near the origin denom→0; coefficient → 1 (L'Hôpital).
    coeff = torch.where(denom < 1e-4, torch.ones_like(denom), alpha / denom)
    return coeff * xs


# ── LorentzEmbedding ──────────────────────────────────────────────────────────

class LorentzEmbedding(nn.Module):
    """Embedding on the Lorentz hyperboloid, projected to the tangent space.

    Pipeline: token_id → space_embed (d-dim) → project_to_hyperboloid (d+1-dim)
              → log_map_origin (d-dim tangent vector).

    Weight-tied attend() maps a hidden state back to logits via the tangent-space
    weight matrix.

    Args:
        num_embeddings: vocabulary size.
        features:       spatial dimension d (output dim equals d, not d+1).
    """

    def __init__(self, num_embeddings: int, features: int):
        super().__init__()
        self.features = features
        self.space_embed = nn.Embedding(num_embeddings, features)
        nn.init.normal_(self.space_embed.weight, std=0.005)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return tangent-space embedding, shape [..., features]."""
        space = self.space_embed(input_ids)
        return _log_map_origin(_project_to_hyperboloid(space))

    def attend(self, x: Tensor) -> Tensor:
        """Weight-tied projection for LM head logits.

        Args:
            x: [..., features] hidden state.

        Returns:
            [..., vocab_size] logits.
        """
        tangent_w = _log_map_origin(_project_to_hyperboloid(self.space_embed.weight))
        return x @ tangent_w.T


# ── HybridEmbedding ───────────────────────────────────────────────────────────

class HybridEmbedding(nn.Module):
    """Concatenated Euclidean + Lorentz embedding.

    Output: [euc_embed(ids) || lor_embed(ids)], shape [..., d_model].

    Args:
        vocab_size:       vocabulary size.
        d_model:          total output dimension (euclidean_dim + lorentz_dim).
        lorentz_fraction: fraction of d_model allocated to the Lorentz channel.
                          lorentz_dim = int(d_model * lorentz_fraction).
                          euclidean_dim = d_model - lorentz_dim.
    """

    def __init__(self, vocab_size: int, d_model: int, lorentz_fraction: float):
        super().__init__()
        self.lorentz_dim  = int(d_model * lorentz_fraction)
        self.euclidean_dim = d_model - self.lorentz_dim
        assert self.euclidean_dim > 0, (
            f"lorentz_fraction={lorentz_fraction} leaves no room for euclidean dims"
        )
        assert self.lorentz_dim > 0, (
            f"lorentz_fraction={lorentz_fraction} leaves no room for lorentz dims"
        )

        self.euc_embed = nn.Embedding(vocab_size, self.euclidean_dim)
        nn.init.normal_(self.euc_embed.weight, std=1.0 / math.sqrt(self.euclidean_dim))

        self.lor_embed = LorentzEmbedding(vocab_size, self.lorentz_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return [euclidean || lorentz] embedding, shape [..., d_model]."""
        return torch.cat(
            [self.euc_embed(input_ids), self.lor_embed(input_ids)],
            dim=-1,
        )

    def attend(self, x: Tensor) -> Tensor:
        """Weight-tied LM head logits, shape [..., vocab_size].

        Splits x into euclidean and lorentz slices, computes logits from each
        weight matrix, and sums them for the final score.
        """
        x_euc = x[..., :self.euclidean_dim]
        x_lor = x[..., self.euclidean_dim:]

        euc_w = self.euc_embed.weight                                       # [V, euc_dim]
        lor_w = _log_map_origin(
            _project_to_hyperboloid(self.lor_embed.space_embed.weight)     # [V, lor_dim]
        )
        # Full vocab weight matrix: [V, d_model]
        w_full = torch.cat([euc_w, lor_w], dim=-1)
        return x @ w_full.T


# ── BigramEmbedding ───────────────────────────────────────────────────────────

_BIGRAM_HASH_A: int = 279470273
_BIGRAM_HASH_B: int = 4294967291


class BigramEmbedding(nn.Module):
    """Hash-based 2-gram embedding with per-layer learned lambda scaling.

    Each token's bigram key is: (A * token_id XOR B * prev_token_id) % hash_vocab.
    This is position-invariant and captures local n-gram statistics without
    an explicit bigram table (hash collision rate ~1/hash_vocab).

    Per-layer lambdas start at 0 (no bigram signal at init) and are learned.
    Use inject() at each layer to add the bigram signal.

    Args:
        vocab_size:  model vocabulary size (unused in hashing, kept for API parity).
        d_model:     embedding dimension.
        n_layers:    total number of layers (prelude + core + coda).
        hash_vocab:  hash table size. Default 49152 keeps collisions low with
                     reasonable memory (~12 MB for d=768 bf16).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        hash_vocab: int = 49152,
    ):
        super().__init__()
        self.hash_vocab = hash_vocab
        self.embed = nn.Embedding(hash_vocab, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        # Per-layer scalar lambdas — init 0 so bigram has no effect at start.
        self.lambdas = nn.Parameter(torch.zeros(n_layers))

    def compute(self, input_ids: Tensor) -> Tensor:
        """Compute bigram embedding for every token position.

        Args:
            input_ids: [B, S] integer token ids.

        Returns:
            [B, S, d_model] bigram embeddings.
        """
        prev = F.pad(input_ids[:, :-1], (1, 0), value=0)   # shift right, pad 0
        hash_ids = (
            _BIGRAM_HASH_A * input_ids ^ _BIGRAM_HASH_B * prev
        ) % self.hash_vocab
        return self.embed(hash_ids)

    def inject(self, x: Tensor, bigram_emb: Tensor, layer_idx: int) -> Tensor:
        """Add scaled bigram signal to the residual stream.

        Args:
            x:          [B, S, d_model] residual stream.
            bigram_emb: [B, S, d_model] output of compute().
            layer_idx:  which lambda scalar to use.

        Returns:
            [B, S, d_model] updated residual stream.
        """
        lam = self.lambdas[layer_idx].to(x.dtype)
        return x + lam * bigram_emb


# ── MORPHEmbedding ────────────────────────────────────────────────────────────

class MORPHEmbedding(nn.Module):
    """Complete MORPH embedding module: HybridEmbedding + BigramEmbedding.

    All features always on:
      - HybridEmbedding provides the input token representation
        (Euclidean || Lorentz channels, always both active).
      - BigramEmbedding provides per-layer bigram injection signals.
      - attend() is the weight-tied LM head.

    Args:
        vocab_size:       vocabulary size.
        d_model:          model hidden dimension.
        lorentz_fraction: fraction of d_model for the Lorentz channel.
        bigram_hash_vocab: hash table size for bigram embeddings.
        n_layers:         total layers (used to size per-layer bigram lambdas).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        lorentz_fraction: float,
        bigram_hash_vocab: int,
        n_layers: int,
    ):
        super().__init__()
        self.d_model = d_model

        self.hybrid = HybridEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            lorentz_fraction=lorentz_fraction,
        )
        self.bigram = BigramEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            hash_vocab=bigram_hash_vocab,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return hybrid (euclidean+lorentz) embedding, shape [B, S, d_model].

        Call this once at the start of the forward pass to get the input
        representation. Bigram injections are applied per-layer via get_bigram()
        + bigram.inject().
        """
        return self.hybrid(input_ids)

    def get_bigram(self, input_ids: Tensor) -> Tensor:
        """Compute bigram embeddings to be injected at each layer.

        Args:
            input_ids: [B, S] integer token ids.

        Returns:
            [B, S, d_model] bigram embeddings. Pass to bigram.inject() at each
            layer with the appropriate layer_idx.
        """
        return self.bigram.compute(input_ids)

    def attend(self, x: Tensor) -> Tensor:
        """Weight-tied LM head: hidden state → vocab logits.

        Args:
            x: [B, S, d_model] final hidden state (after norm).

        Returns:
            [B, S, vocab_size] logits.
        """
        return self.hybrid.attend(x)
