"""Block-sparse linear layer with dynamic CMS topology updates.

CMSBlockLinear is a drop-in replacement for nn.Linear that uses:
- Dense weight storage pre-compact (cuBLAS speed, tile-level CMS scoring)
- Block-ELL sparse weight storage post-compact (FLOP savings, Triton kernels)
- Gradient-based importance scoring for topology decisions
- Periodic topology updates (prune/grow blocks) via CMS Level 2

Pre-compact: weights stored as standard [out, in] matrix (F.linear → cuBLAS).
Pruning zeros tile regions in the dense matrix. No sparse overhead.
Post-compact: weights rebuilt into Block-ELL [R, K_active, B, B] format.
Triton kernel skips dead macro tiles for actual FLOP savings.

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .block_ell import BlockELLConfig
from titans_core.opt.topology_scorer import TopologyScorer, compute_gradient_frobenius_norms

# Check if Triton is available
try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@dataclass
class TopologyStats:
    """Statistics about current topology state.

    Attributes:
        density: Actual density (K/C)
        avg_block_score: Mean gradient EMA across blocks
        avg_block_age: Mean age in topology steps
        column_entropy: Entropy of column usage (0-1 normalized)
        num_blocks: Total active blocks (R * K)
        mean_coherence: Mean gradient coherence EMA (FR-027)
        crystallized_count: Number of crystallized blocks
        avg_swap_count: Mean swap count per block (Fix #5)
        blocks_in_cooldown: Number of blocks currently in cooldown (Fix #5)
        max_swap_count: Maximum swap count across all blocks (Fix #5)
    """

    density: float
    avg_block_score: float
    avg_block_age: float
    column_entropy: float
    num_blocks: int
    mean_coherence: float = 0.0
    crystallized_count: int = 0
    avg_swap_count: float = 0.0
    blocks_in_cooldown: int = 0
    max_swap_count: int = 0


@dataclass
class TopologyDecisionResult:
    """Result of a topology step.

    Attributes:
        num_swaps: Blocks swapped this step
        swap_rate: num_swaps / total_blocks
        pruned_positions: List of (row, slot) pruned
        grown_columns: List of new column indices grown
        consolidation_triggered: Whether consolidation ran this step (T030)
        crystallized_count: Blocks newly crystallized this step (T030)
        pressure_ratio: Gradient pressure ratio for diagnostics (T031)
    """

    num_swaps: int
    swap_rate: float
    pruned_positions: List[Tuple[int, int]]
    grown_columns: List[int]
    consolidation_triggered: bool = False
    crystallized_count: int = 0
    pressure_ratio: float = 1.0


class CMSBlockLinear(nn.Module):
    """Block-sparse linear layer with dynamic topology via CMS Level 2 updates.

    Drop-in replacement for nn.Linear with:
    - Block-ELL sparse weight storage
    - Gradient-based importance scoring
    - Periodic topology updates (prune/grow blocks)

    Args:
        in_features: Input dimension (must be divisible by tile_size)
        out_features: Output dimension (must be divisible by tile_size)
        tile_size: Block size (default 16 for WMMA compatibility)
        density: Fraction of active blocks per row (0.1 to 1.0)
        bias: Include bias term (default True)
        score_ema_alpha: EMA momentum for gradient scores (default 0.95)
        swap_threshold: Required improvement ratio for topology swaps (default 1.5)
        exploration_epsilon: Random swap probability for diversity (default 0.05)
        use_age_protection: Apply age-based score bonus to prevent churn (default True)
        device: Target device
        dtype: Parameter dtype

    Raises:
        ValueError: If dimensions not divisible by tile_size
        ValueError: If density not in [0.1, 1.0]

    Example:
        >>> layer = CMSBlockLinear(640, 2560, tile_size=16, density=0.5)
        >>> x = torch.randn(32, 640)
        >>> y = layer(x)  # [32, 2560]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tile_size: int = 16,
        density: float = 0.5,
        bias: bool = True,
        score_ema_alpha: float = 0.95,
        swap_threshold: float = 2.5,
        exploration_epsilon: float = 0.05,
        use_age_protection: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # === CMS Topology Evolution v2 Parameters ===
        usage_decay: float = 0.99,
        reserve_threshold: float = 0.3,
        reserve_weight: float = 0.5,
        protection_weights: Tuple[float, float, float] = (0.3, 0.3, 0.4),
        row_swap_cap_fraction: float = 0.2,
        layer_swap_budget_fraction: float = 0.05,
        pressure_high_threshold: float = 1.5,
        pressure_low_threshold: float = 0.7,
        consolidation_interval: int = 1000,
        crystallize_age_threshold: int = 1000,
        crystallize_coherence_threshold: float = 0.7,
        crystallize_protection_multiplier: float = 10.0,
        disable_topology_updates: bool = False,
        # === Topology Warm-up ===
        topology_warmup_steps: int = 0,  # Global steps before first topology swap. 0 = disabled.
        # === Fix #5: Churn Protection Parameters ===
        swap_cooldown_steps: int = 500,  # Can't swap again for N global steps
        churn_penalty_factor: float = 0.1,  # Penalty per swap count
        coherence_decay_on_swap: float = 0.7,  # Preserve 70% of coherence on swap
        score_history_decay_on_swap: float = 0.5,  # Decay old scores but don't discard
    ) -> None:
        """Initialize block-sparse linear layer."""
        super().__init__()

        # Validate inputs
        if in_features % tile_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by tile_size ({tile_size})"
            )
        if out_features % tile_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by tile_size ({tile_size})"
            )
        if not (0.1 <= density <= 1.0):
            raise ValueError(f"density ({density}) must be in [0.1, 1.0]")

        # Validate protection_weights sum to 1.0
        weight_sum = sum(protection_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"protection_weights must sum to 1.0, got {protection_weights} (sum={weight_sum})"
            )

        # Core dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        self.density = density
        self.score_ema_alpha = score_ema_alpha
        self.swap_threshold = swap_threshold
        self.exploration_epsilon = exploration_epsilon
        self.use_age_protection = use_age_protection

        # CMS Topology Evolution v2 parameters
        self.usage_decay = usage_decay
        self.reserve_threshold = reserve_threshold
        self.reserve_weight = reserve_weight
        self.protection_weights = protection_weights
        self.row_swap_cap_fraction = row_swap_cap_fraction
        self.layer_swap_budget_fraction = layer_swap_budget_fraction
        self.pressure_high_threshold = pressure_high_threshold
        self.pressure_low_threshold = pressure_low_threshold
        self.consolidation_interval = consolidation_interval
        self.crystallize_age_threshold = crystallize_age_threshold
        self.crystallize_coherence_threshold = crystallize_coherence_threshold  # DEPRECATED: coherence removed
        self.crystallize_protection_multiplier = crystallize_protection_multiplier
        self.disable_topology_updates = disable_topology_updates
        self.topology_warmup_steps = topology_warmup_steps

        # Fix #5: Churn protection parameters
        self.swap_cooldown_steps = swap_cooldown_steps
        self.churn_penalty_factor = churn_penalty_factor
        self.coherence_decay_on_swap = coherence_decay_on_swap  # DEPRECATED: coherence removed
        self.score_history_decay_on_swap = score_history_decay_on_swap

        # Derived dimensions
        self.R = out_features // tile_size  # output block-rows
        self.C = in_features // tile_size  # input block-columns
        self.K = max(1, int(self.C * density))  # active blocks per row

        # Block-ELL config
        self._block_ell_config = BlockELLConfig(R=self.R, C=self.C, K=self.K, B=tile_size)

        # Dense mode: store weights as standard [out, in] matrix for cuBLAS speed.
        # After compact(), transitions to Block-ELL [R, K_active, B, B] sparse format.
        self._dense_mode = True
        self._ternary_mode = False

        # Dense weight parameter — same as nn.Linear
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Block-ELL state — created at compact() time, not at init
        # self.values will be set when transitioning to sparse mode
        self.register_buffer(
            "col_indices",
            torch.zeros(self.R, self.K, dtype=torch.int32, device=device),
        )

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # === Scoring State (buffers, not parameters) ===

        # Gradient importance EMA [R, K]
        self.register_buffer(
            "block_score_ema",
            torch.zeros(self.R, self.K, device=device, dtype=dtype or torch.float32),
        )

        # Input activation norms accumulated [C]
        self.register_buffer(
            "activation_norm_acc",
            torch.zeros(self.C, device=device, dtype=dtype or torch.float32),
        )

        # Output error norms accumulated [R]
        self.register_buffer(
            "error_norm_acc",
            torch.zeros(self.R, device=device, dtype=dtype or torch.float32),
        )

        # Block age (steps since creation) [R, K]
        self.register_buffer(
            "block_age",
            torch.zeros(self.R, self.K, dtype=torch.int32, device=device),
        )

        # Score snapshot: preserved for visualization before reset in topology_step()
        # This allows checkpoints to have meaningful score data for analysis
        self.register_buffer(
            "_score_snapshot",
            torch.zeros(self.R, self.K, device=device, dtype=dtype or torch.float32),
        )

        # === CMS Topology Evolution v2 Buffers ===

        # Column usage count [C] - tracks how often each input column is connected
        # Updated: decay by 0.99 in score_step(); increment from col_indices in topology_step()
        self.register_buffer(
            "col_usage_count",
            torch.zeros(self.C, device=device, dtype=torch.float32),
        )

        # Score history ring buffer [R, K, 10] - for stability/variance computation
        self.register_buffer(
            "score_history",
            torch.zeros(self.R, self.K, 10, device=device, dtype=dtype or torch.float32),
        )

        # Crystallized mask [R, K] - tracks blocks protected from replacement
        self.register_buffer(
            "crystallized_mask",
            torch.zeros(self.R, self.K, device=device, dtype=torch.bool),
        )

        # Historical score EMA [R, K] - long-term baseline for pressure ratio computation
        # Updated with decay 0.999 in score_step()
        self.register_buffer(
            "block_score_historical_ema",
            torch.zeros(self.R, self.K, device=device, dtype=dtype or torch.float32),
        )

        # === Fix #5: Churn Protection Buffers ===

        # Last swap step [R, K] - tracks global_step when each block was last swapped
        # Used for cooldown period enforcement
        self.register_buffer(
            "last_swap_step",
            torch.zeros(self.R, self.K, dtype=torch.int32, device=device),
        )

        # Swap count [R, K] - tracks how many times each block position has been swapped
        # Used for churn penalty (blocks that churn often get penalized)
        self.register_buffer(
            "swap_count",
            torch.zeros(self.R, self.K, dtype=torch.int32, device=device),
        )

        # Score history ring buffer index pointer (not a buffer, just an int)
        self._score_history_idx: int = 0

        # Topology step counter for consolidation interval tracking
        self._topology_step_count: int = 0

        # Step counter for accumulator normalization
        self._acc_steps: int = 0

        # Topology history for monitoring (T077)
        # Stores (before_col_indices, after_col_indices) tuples
        self._topology_history: List[Tuple[Tensor, Tensor]] = []
        self._topology_history_max_size: int = 10  # Limit memory usage

        # Swap rate tracking (T074)
        self._swap_rate_history: List[float] = []
        self._swap_rate_history_max_size: int = 100  # Rolling window

        # Track last pruned positions for optimizer state reset (Fix 1)
        self._last_pruned_positions: List[Tuple[int, int]] = []

        # Column index cache for optimized dx backward kernel
        # The cache is invalidated when topology changes (after topology_step())
        # Cache format: (pairs, counts, max_pairs) or None
        self._column_index_cache: Optional[Tuple[Tensor, Tensor, int]] = None
        self._column_index_valid: bool = False

        # Initialize weights and topology
        self._reset_parameters()
        self._initialize_topology()

        # Register hooks for activation and gradient capture (T029, T030)
        self.register_forward_hook(self._activation_hook)
        self.register_full_backward_hook(self._gradient_hook)

    def _reset_parameters(self) -> None:
        """Initialize weight parameters matching nn.Linear's kaiming_uniform(a=sqrt(5))."""
        if self._dense_mode:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            fan_in = self.K * self.tile_size
            gain = nn.init.calculate_gain("relu")
            std = gain / (fan_in**0.5)
            nn.init.normal_(self.values, mean=0.0, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def enable_ternary(self) -> None:
        """Transition from dense → dense+ternary (Phase 1 → Phase 2).
        Shadow weight IS self.weight — zero extra memory. STE passes gradients through."""
        assert self._dense_mode, "Ternary only supported in dense mode"
        self._ternary_mode = True
        B = self.tile_size
        w_tiles = self.weight.data.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)
        tile_means = w_tiles.abs().mean(dim=(2, 3))  # [R, C]
        self.register_buffer("ternary_scale", tile_means.clamp(min=1e-6))

    def _ternary_ste(self, w: Tensor) -> Tensor:
        """Quantize to {-1, 0, +1} × per-tile scale with straight-through estimator."""
        B = self.tile_size
        w_tiles = w.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)  # [R, C, B, B]
        scale = self.ternary_scale.unsqueeze(-1).unsqueeze(-1)  # [R, C, 1, 1]
        w_scaled = w_tiles / scale.clamp(min=1e-6)
        w_ternary = torch.sign(w_scaled) * (w_scaled.abs() > 0.5).float()
        result = scale * (w_scaled + (w_ternary - w_scaled).detach())
        return result.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    def update_ternary_scales(self) -> None:
        """Recompute per-tile scales from current shadow weights."""
        if not self._ternary_mode:
            return
        B = self.tile_size
        w_tiles = self.weight.data.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)
        self.ternary_scale = w_tiles.abs().mean(dim=(2, 3)).clamp(min=1e-6)

    def _initialize_topology(self) -> None:
        """Initialize column indices. Dense mode: identity. Sparse mode: random."""
        if self._dense_mode:
            for r in range(self.R):
                self.col_indices[r] = torch.arange(self.K, dtype=torch.int32,
                                                    device=self.col_indices.device)
        else:
            for r in range(self.R):
                perm = torch.randperm(self.C, device=self.col_indices.device)
                self.col_indices[r] = perm[: self.K].to(torch.int32)
        self._column_index_valid = False

    def init_random_structured_topology(self) -> None:
        """T054 (FR-029): Initialize random topology ensuring each column appears in at least R/C rows.

        This method generates a structured random topology that distributes columns
        evenly across rows. Unlike pure random initialization which may leave some
        columns unused, this ensures balanced column coverage.

        Algorithm:
        1. Calculate min_occurrences = max(1, R // C) per column
        2. Create a pool of column indices where each column appears at least min_occurrences times
        3. Shuffle the pool and distribute to rows, ensuring each row has K unique columns

        The result is a random topology with guaranteed column coverage, useful for
        baseline comparison experiments to isolate the value of learned topology.

        Contract:
            - Each column appears in at least floor(R/C) rows
            - Each row has exactly K unique columns
            - All column indices are in valid range [0, C)
            - Column index cache is invalidated
        """
        with torch.no_grad():
            device = self.col_indices.device

            # Calculate minimum occurrences per column (FR-029: R/C rows)
            min_occurrences = max(1, self.R // self.C)

            # Total slots available across all rows
            total_slots = self.R * self.K

            # Total slots per column if distributed evenly
            # Each column should appear approximately R*K/C times on average
            avg_per_col = total_slots // self.C
            remainder = total_slots % self.C

            # Build a list of column indices where each column appears
            # at least min_occurrences times, distributed as evenly as possible
            col_assignments = []

            # First, ensure minimum coverage for all columns
            for col in range(self.C):
                # Base count ensures even distribution
                # Some columns get one extra to account for remainder
                count = avg_per_col + (1 if col < remainder else 0)
                col_assignments.extend([col] * count)

            # Shuffle the assignments for randomness
            col_assignments = torch.tensor(col_assignments, dtype=torch.int32, device=device)
            perm = torch.randperm(len(col_assignments), device=device)
            col_assignments = col_assignments[perm]

            # Now assign to rows, ensuring each row gets K unique columns
            # Use a greedy approach: for each row, pick K columns from the pool
            # that haven't been assigned to this row yet
            col_pool = col_assignments.tolist()
            pool_idx = 0

            for r in range(self.R):
                row_cols = set()

                # Try to pick K unique columns for this row
                attempts = 0
                max_attempts = len(col_pool)  # Prevent infinite loop

                while len(row_cols) < self.K and attempts < max_attempts:
                    col = col_pool[pool_idx % len(col_pool)]
                    if col not in row_cols:
                        row_cols.add(col)
                    pool_idx += 1
                    attempts += 1

                    # If we've exhausted the pool and still need columns,
                    # just pick from available columns not in this row
                    if pool_idx >= len(col_pool) and len(row_cols) < self.K:
                        available = [c for c in range(self.C) if c not in row_cols]
                        while len(row_cols) < self.K and available:
                            row_cols.add(available.pop())

                # Convert to tensor and assign
                row_tensor = torch.tensor(list(row_cols), dtype=torch.int32, device=device)
                self.col_indices[r] = row_tensor[:self.K]

            # Invalidate column index cache when topology changes
            self._column_index_valid = False

    def _ensure_column_index(self) -> Optional[Tuple[Tensor, Tensor, int]]:
        """Ensure column index cache is valid, rebuilding if needed.

        The column index is a precomputed reverse mapping from input columns
        to (r, k) pairs. It enables the optimized column-parallel dx backward
        kernel which is 7.3x faster than the vectorized implementation.

        The cache is invalidated when:
        - topology_step() changes col_indices
        - _initialize_topology() is called
        - The layer is first created

        Returns:
            Tuple of (pairs, counts, max_pairs) or None if caching is disabled
        """
        if self._column_index_valid and self._column_index_cache is not None:
            return self._column_index_cache

        # Build the column index
        from titans_core.kernels.block_ell_backward import build_column_index

        pairs, counts, max_pairs = build_column_index(self.col_indices, self.C)
        self._column_index_cache = (pairs, counts, max_pairs)
        self._column_index_valid = True

        return self._column_index_cache

    def invalidate_column_index_cache(self) -> None:
        """Invalidate the column index cache.

        Call this after any operation that changes col_indices, such as
        topology_step(). The cache will be rebuilt on the next backward pass.
        """
        self._column_index_valid = False
        self._column_index_cache = None

    @property
    def _use_triton_kernel(self) -> bool:
        """T050: Detect whether to use Triton kernels for forward/backward.

        Triton kernels are used when:
        1. Triton is installed and available
        2. Tensors are on CUDA device
        3. Tile size is compatible (8, 16, 32, or 64)

        Returns:
            True if Triton kernels should be used
        """
        return TRITON_AVAILABLE and self.values.is_cuda and self.tile_size in (8, 16, 32, 64)

    def _activation_hook(
        self, module: nn.Module, input: Tuple[Tensor, ...], output: Tensor
    ) -> None:
        """T029: Capture input activation norms per block-column.

        Computes L2 norm of input activations for each block-column and accumulates
        into activation_norm_acc buffer. Only active during training.

        Args:
            module: The module (self)
            input: Tuple containing input tensor(s)
            output: Output tensor (not used)
        """
        if not self.training:
            return

        x = input[0]  # [batch, in_features] or [batch, seq, in_features]
        B = self.tile_size

        # Handle 2D vs 3D input
        if x.dim() == 2:
            # [batch, in_features] -> [batch, 1, in_features]
            x = x.unsqueeze(1)

        # x is now [batch, seq, in_features]
        batch_size, seq_len, _ = x.shape

        # Reshape to block view: [batch, seq, C, B]
        x_blocks = x.view(batch_size, seq_len, self.C, B)

        # Compute L2 norm per block-column: sqrt(sum of squared elements)
        # First flatten batch and seq: [batch * seq, C, B]
        x_flat = x_blocks.view(-1, self.C, B)

        # L2 norm per column: [C]
        # Sum over batch and feature dimensions, then sqrt
        col_norms = torch.sqrt(torch.sum(x_flat * x_flat, dim=(0, 2)))

        # Accumulate into buffer (detach to avoid autograd issues)
        with torch.no_grad():
            self.activation_norm_acc = self.activation_norm_acc + col_norms.detach()

    def _gradient_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[Tensor], ...],
        grad_output: Tuple[Tensor, ...],
    ) -> None:
        """T030: Capture output gradient norms per block-row.

        Computes L2 norm of output gradients for each block-row and accumulates
        into error_norm_acc buffer. Only active during training.

        Args:
            module: The module (self)
            grad_input: Gradients with respect to inputs (not used)
            grad_output: Tuple containing gradient tensor(s) with respect to output
        """
        if not self.training:
            return

        grad = grad_output[0]  # [batch, out_features] or [batch, seq, out_features]
        if grad is None:
            return

        B = self.tile_size

        # Handle 2D vs 3D input
        if grad.dim() == 2:
            # [batch, out_features] -> [batch, 1, out_features]
            grad = grad.unsqueeze(1)

        # grad is now [batch, seq, out_features]
        batch_size, seq_len, _ = grad.shape

        # Reshape to block view: [batch, seq, R, B]
        grad_blocks = grad.view(batch_size, seq_len, self.R, B)

        # Compute L2 norm per block-row: sqrt(sum of squared elements)
        # First flatten batch and seq: [batch * seq, R, B]
        grad_flat = grad_blocks.view(-1, self.R, B)

        # L2 norm per row: [R]
        # Sum over batch and feature dimensions, then sqrt
        row_norms = torch.sqrt(torch.sum(grad_flat * grad_flat, dim=(0, 2)))

        # Accumulate into buffer (detach to avoid autograd issues)
        with torch.no_grad():
            self.error_norm_acc = self.error_norm_acc + row_norms.detach()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: dense (cuBLAS) pre-compact, Block-ELL sparse post-compact.

        Pre-compact (dense mode): F.linear(x, weight, bias) → cuBLAS GEMM.
        Post-compact (sparse mode): Triton Block-ELL kernel or PyTorch reference.

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]

        Returns:
            Output tensor with same batch/seq dims, out_features last
        """
        if self._dense_mode:
            w = self._ternary_ste(self.weight) if self._ternary_mode else self.weight
            return F.linear(x, w, self.bias)

        # Sparse mode: Triton or reference
        if self._use_triton_kernel and not hasattr(self, "spartan_active_weights"):
            return self._forward_triton(x)
        else:
            return self._forward_reference(x)

    def _forward_triton(self, x: Tensor) -> Tensor:
        """Forward using Triton kernels with autograd support.

        Uses block_ell_forward_autograd which wraps the Triton kernel in
        torch.autograd.Function for proper gradient computation.

        If training mode, ensures the column index cache is valid for the
        optimized column-parallel dx backward kernel (7.3x faster).

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]

        Returns:
            Output tensor
        """
        from titans_core.kernels.block_ell_forward import block_ell_forward_autograd

        # Get column index for optimized backward pass (only during training)
        column_index = None
        if self.training:
            column_index = self._ensure_column_index()

        return block_ell_forward_autograd(
            x=x,
            values=self.values,
            col_indices=self.col_indices,
            bias=self.bias,
            use_triton=True,
            column_index=column_index,
        )

    def _forward_reference(self, x: Tensor) -> Tensor:
        """Forward using vectorized PyTorch reference implementation.

        Fully batched — no Python loops over R rows. Uses advanced indexing
        and a single einsum to compute all block-sparse matmuls simultaneously.

        Supports optional spartan_active_weights [R, K] for differentiable
        column selection (STE: values are 1.0 in forward, gradients flow in backward).

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]

        Returns:
            Output tensor
        """
        # Handle 2D vs 3D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = x.shape
        B = self.tile_size

        # Reshape input to block view: [batch, seq, C, B]
        x_blocks = x.view(batch_size, seq_len, self.C, B)

        # Gather all input blocks at once: col_indices is [R, K]
        # x_blocks[:, :, col_indices, :] → [batch, seq, R, K, B]
        all_cols = self.col_indices.long()  # [R, K]
        input_gathered = x_blocks[:, :, all_cols, :]  # [batch, seq, R, K, B]

        # Batched block-sparse matmul via single einsum:
        # input_gathered: [batch, seq, R, K, B_in]
        # self.values:    [R, K, B_out, B_in]
        # → block_outputs: [batch, seq, R, K, B_out]
        block_outputs = torch.einsum("bsrki,rkoi->bsrko", input_gathered, self.values)

        # Spartan: scale tile outputs by differentiable active_weights
        # for gradient flow to Spartan scores. STE: aw=1.0 in forward,
        # so numerics are unchanged; backward chain exists.
        aw = getattr(self, "spartan_active_weights", None)
        if aw is not None:
            block_outputs = block_outputs * aw[None, None, :, :, None]

        # Sum over K tiles per row: [batch, seq, R, B]
        output = block_outputs.sum(dim=3)

        # Reshape: [batch, seq, R, B] → [batch, seq, out_features]
        output = output.reshape(batch_size, seq_len, self.out_features)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        # Squeeze back if input was 2D
        if squeeze_output:
            output = output.squeeze(1)  # [batch, out_features]

        return output

    def accumulate_scores(self) -> None:
        """T031: Accumulate gradient statistics for importance scoring.

        Call after backward() each step. Updates:
        - block_score_ema: EMA of gradient Frobenius norms
        - activation_norm_acc: Accumulated input norms (requires hook)
        - error_norm_acc: Accumulated output error norms (requires hook)

        Dense mode: reshapes weight.grad [out, in] into tile view [R, K, B, B].
        Sparse mode: uses values.grad [R, K, B, B] directly.

        Contract:
            - Safe to call even if grad is None (no-op)
            - Accumulates into existing EMA (doesn't reset)
        """
        if self._dense_mode:
            if self.weight.grad is None:
                return
            B = self.tile_size
            grad_tiles = self.weight.grad.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)
            grad_norms = compute_gradient_frobenius_norms(grad_tiles)
        else:
            if self.values.grad is None:
                return
            grad_norms = compute_gradient_frobenius_norms(self.values.grad)

        scorer = TopologyScorer(self.R, self.C, self.K, ema_alpha=self.score_ema_alpha)

        with torch.no_grad():
            self.block_score_ema = scorer.update_gradient_ema(grad_norms, self.block_score_ema)

        self._acc_steps += 1

    def _compute_stability(self) -> Tensor:
        """T019: Compute stability from score history variance.

        Stability measures how consistent a block's gradient importance has been
        over the last 10 score_step() calls. Blocks with stable scores are more
        valuable and should be protected from replacement.

        Formula (FR-008):
            variance = score_history.var(dim=-1)  # Variance across 10-element ring buffer
            stability = 1.0 / (1.0 + variance)

        Returns:
            stability: Tensor [R, K] with values in (0, 1]
                - High variance -> low stability (approaching 0)
                - Low variance -> high stability (approaching 1)
                - Zero variance -> stability = 1.0

        Edge cases:
            - All zeros in history: variance=0, stability=1.0 (neutral/protected)
            - Very high variance: stability approaches 0
        """
        with torch.no_grad():
            # Compute variance across the 10-element ring buffer (dim=-1)
            # Use unbiased=False for population variance (divide by N, not N-1)
            # This matches the formula in spec: variance = score_history.var(dim=-1)
            # and ensures stability=1.0 when all values are identical
            # score_history shape: [R, K, 10]
            variance = self.score_history.var(dim=-1, unbiased=False)  # [R, K]

            # Stability is inverse of variance (bounded by 1/(1+var))
            # High variance = low stability, low variance = high stability
            stability = 1.0 / (1.0 + variance)  # [R, K]

            return stability

    def _compute_protection_scores(self) -> Tensor:
        """T020: Compute multi-factor protection scores for all blocks.

        Protection scores determine how resistant blocks are to replacement.
        Higher protection = harder to replace (score gets boosted).

        Formula (coherence removed, weights redistributed):
            protection = w_age * min(age/500, 1.0) + w_stability * stability

        Where:
            - w_age = protection_weights[0] + protection_weights[2] * 0.5
            - w_stability = protection_weights[1] + protection_weights[2] * 0.5
            - age: block_age (in topology steps)
            - stability: 1/(1+variance) from _compute_stability()

        Age uses LINEAR RAMP to 500 (not sigmoid) to avoid saturation issues.
        Sigmoid saturates around age ~300, losing discrimination for older blocks.

        Returns:
            protection: Tensor [R, K] with values in [0, 1]
                - 0.0: No protection (new block, unstable)
                - 1.0: Maximum protection (old, stable)

        Edge cases:
            - age=0: age_term = 0 (no age protection)
            - variance=0: stability = 1.0 (max stability protection)
        """
        with torch.no_grad():
            # Coherence removed (observability proved it's useless ~0.5)
            # Redistribute weights: age and stability split the coherence weight
            w_age_orig, w_stability_orig, w_coherence_orig = self.protection_weights
            w_age = w_age_orig + w_coherence_orig * 0.5
            w_stability = w_stability_orig + w_coherence_orig * 0.5

            # 1. Age term: LINEAR RAMP from 0 to 1 over 500 topology steps
            # min(age/500, 1.0) - clamped at 1.0 for ages >= 500
            age_term = torch.clamp(self.block_age.float() / 500.0, min=0.0, max=1.0)  # [R, K]

            # 2. Stability term: from score history variance
            stability = self._compute_stability()  # [R, K]

            # Combine with protection weights (coherence removed)
            protection = w_age * age_term + w_stability * stability

            # T032 (FR-022): Apply crystallized protection multiplier
            # Crystallized blocks get 10x protection boost, making them much harder to replace.
            # With swap_threshold of 2.5, this means candidates need ~25x better scores to replace.
            protection = torch.where(
                self.crystallized_mask,
                protection * self.crystallize_protection_multiplier,
                protection
            )

            return protection  # [R, K], crystallized blocks have values up to 10.0

    def _compute_reserved_columns(self) -> Tensor:
        """T036 (FR-010): Compute reserved column mask for capacity reservation.

        Reserved columns are those with usage below reserve_threshold * mean_usage.
        These columns are considered underutilized and get bonus scoring during
        topology decisions to encourage new tasks to use them rather than
        fighting for established pathways.

        Formula (FR-010):
            reserved = col_usage_count < (reserve_threshold * mean_usage)

        Returns:
            Boolean mask [C] where True indicates reserved (underutilized) columns.

        Edge cases:
            - All zeros: mean=0, threshold=0, 0<0=False -> no columns reserved
              (intentional: at start, no established patterns to protect)
            - Uniform usage: all values equal -> none reserved (all above threshold)
            - Boundary: usage exactly at threshold -> NOT reserved (strict <)
        """
        with torch.no_grad():
            mean_usage = self.col_usage_count.mean()
            threshold = self.reserve_threshold * mean_usage
            reserved = self.col_usage_count < threshold
            return reserved

    def consolidation_step(self) -> dict:
        """T029 (FR-020 to FR-024b): Perform era-level consolidation.

        Consolidation crystallizes stable blocks and de-crystallizes dead ones:
        - Crystallize: blocks with age > crystallize_age_threshold AND stability > 0.7
        - De-crystallize: crystallized blocks with stability < 0.3 (dead block recovery)

        Called from topology_step() every consolidation_interval steps, or early
        when pressure_ratio > 2.0 (distribution shift protection).

        Returns:
            dict with keys:
                - crystallized_count: Number of blocks newly crystallized this step
                - decrystallized_count: Number of blocks de-crystallized this step
                - total_crystallized: Total crystallized blocks after this step
        """
        with torch.no_grad():
            # Current state before changes
            was_crystallized = self.crystallized_mask.clone()

            # Compute stability for crystallization criteria (coherence removed)
            stability = self._compute_stability()  # [R, K]

            # FR-021: Crystallize blocks meeting criteria
            # age > crystallize_age_threshold AND stability > 0.7
            age_meets = self.block_age >= self.crystallize_age_threshold
            stability_meets = stability > self.crystallize_coherence_threshold  # repurposed from coherence threshold

            # New crystallizations: meets criteria AND not already crystallized
            should_crystallize = age_meets & stability_meets
            new_crystallizations = should_crystallize & ~was_crystallized

            # FR-024b: De-crystallize dead blocks (stability < 0.3)
            # These are blocks that were crystallized but have become unstable
            stability_dead = stability < 0.3
            should_decrystallize = was_crystallized & stability_dead

            # Apply updates to crystallized_mask (FR-024)
            # Add new crystallizations
            self.crystallized_mask = self.crystallized_mask | new_crystallizations
            # Remove de-crystallizations
            self.crystallized_mask = self.crystallized_mask & ~should_decrystallize

            # Compute stats
            crystallized_count = new_crystallizations.sum().item()
            decrystallized_count = should_decrystallize.sum().item()
            total_crystallized = self.crystallized_mask.sum().item()

        return {
            "crystallized_count": int(crystallized_count),
            "decrystallized_count": int(decrystallized_count),
            "total_crystallized": int(total_crystallized),
        }

    def _compute_pressure_ratio(self) -> float:
        """T031 (FR-018): Compute pressure ratio for adaptive exploration and early consolidation.

        Pressure ratio indicates whether current gradient activity is higher (>1) or
        lower (<1) than historical baseline. High pressure (>2.0) indicates potential
        distribution shift requiring protective consolidation.

        Formula (FR-018):
            pressure_ratio = recent_mean / (historical_mean + 1e-8)

        Returns:
            Pressure ratio as float. Returns 1.0 if insufficient data (first 100 topology steps).
        """
        with torch.no_grad():
            # Skip adaptive pressure for first 100 topology steps (FR-019b bootstrap)
            if self._topology_step_count < 100:
                return 1.0

            recent_mean = self.block_score_ema.mean().item()
            historical_mean = self.block_score_historical_ema.mean().item()

            # Compute pressure ratio with epsilon to avoid division by zero
            pressure_ratio = recent_mean / (historical_mean + 1e-8)

            return pressure_ratio

    def score_step(self) -> None:
        """T032: Level 1 update: normalize accumulators and increment ages.

        Call every ~10 training steps. Actions:
        - Decay col_usage_count by usage_decay (FR-002)
        - Normalize activation_norm_acc by step count
        - Normalize error_norm_acc by step count
        - Increment block_age for all active blocks
        - Update block_score_historical_ema (FR-017)
        - Update score_history ring buffer (FR-009)
        - Reset step counter

        Contract:
            - Does NOT reset block_score_ema (kept for Level 2)
            - Does NOT change topology
        """
        with torch.no_grad():
            # T009 (FR-002): Decay column usage counts by 1% each score_step
            self.col_usage_count.mul_(self.usage_decay)

            if self._acc_steps > 0:
                # Normalize accumulated norms by step count
                self.activation_norm_acc = self.activation_norm_acc / self._acc_steps
                self.error_norm_acc = self.error_norm_acc / self._acc_steps

            # Increment block ages
            self.block_age = self.block_age + 1

            # T012 (FR-017): Update historical score EMA with longer decay (0.999)
            # This provides a baseline for pressure ratio computation
            self.block_score_historical_ema.mul_(0.999).add_(self.block_score_ema, alpha=0.001)

            # T013 (FR-009): Update score history ring buffer using index pointer
            # Use index pointer instead of torch.roll() to avoid unnecessary copies
            self.score_history[:, :, self._score_history_idx] = self.block_score_ema
            self._score_history_idx = (self._score_history_idx + 1) % 10

        # Reset step counter (but NOT score EMA - that's kept for Level 2)
        self._acc_steps = 0

    def sync_topology_scores(self) -> None:
        """T082/T059: Synchronize topology scores across DDP ranks.

        All-reduces scoring buffers across all ranks to ensure consistent
        topology decisions. Different buffers use different sync strategies:

        Synced (mean/average):
            - block_score_ema [R, K]: Gradient importance EMA
            - activation_norm_acc [C]: Input activation norms
            - error_norm_acc [R]: Output error norms
            - col_usage_count [C]: Column usage tracking (FR-002)
            - block_score_historical_ema [R, K]: Long-term baseline (FR-017)
            - score_history [R, K, 10]: Rolling history for stability (FR-009)

        Synced (logical OR via MAX):
            - crystallized_mask [R, K]: Block protection status (FR-021)

        Only performs sync if torch.distributed is initialized.

        Contract:
            - No-op if not in distributed training
            - Uses ReduceOp.SUM followed by division for averaging (mean buffers)
            - Uses ReduceOp.MAX for boolean masks (logical OR)
            - Safe to call multiple times
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()

        # All-reduce the scoring buffers
        with torch.no_grad():
            # === Original buffers (T082) ===

            # block_score_ema [R, K]
            dist.all_reduce(self.block_score_ema, op=dist.ReduceOp.SUM)
            self.block_score_ema.div_(world_size)

            # activation_norm_acc [C]
            dist.all_reduce(self.activation_norm_acc, op=dist.ReduceOp.SUM)
            self.activation_norm_acc.div_(world_size)

            # error_norm_acc [R]
            dist.all_reduce(self.error_norm_acc, op=dist.ReduceOp.SUM)
            self.error_norm_acc.div_(world_size)

            # === T059: CMS Topology Evolution v2 buffers ===

            # crystallized_mask [R, K] - logical OR via MAX
            # Must be consistent across ranks for topology decisions (FR-021)
            # Convert to float, MAX across ranks, convert back to bool
            crystallized_float = self.crystallized_mask.float()
            dist.all_reduce(crystallized_float, op=dist.ReduceOp.MAX)
            self.crystallized_mask.copy_(crystallized_float > 0.5)

            # col_usage_count [C] - mean (FR-002)
            # Reserved column computation needs same view across ranks
            dist.all_reduce(self.col_usage_count, op=dist.ReduceOp.SUM)
            self.col_usage_count.div_(world_size)

            # block_score_historical_ema [R, K] - mean (FR-017)
            # Long-term baseline needs consistent view for pressure ratio
            dist.all_reduce(self.block_score_historical_ema, op=dist.ReduceOp.SUM)
            self.block_score_historical_ema.div_(world_size)

            # score_history [R, K, 10] - mean (FR-009)
            # Stability computation needs consistent view across ranks
            dist.all_reduce(self.score_history, op=dist.ReduceOp.SUM)
            self.score_history.div_(world_size)

            # === Fix #5: Churn Protection buffers ===

            # last_swap_step [R, K] - MAX (use most recent swap across ranks)
            # This ensures cooldown is enforced even if swap happened on different rank
            dist.all_reduce(self.last_swap_step, op=dist.ReduceOp.MAX)

            # swap_count [R, K] - MAX (use highest count to be conservative)
            # Higher counts = more protection, so MAX is the safe choice
            dist.all_reduce(self.swap_count, op=dist.ReduceOp.MAX)

            # NOTE: coherence buffers removed (proved useless ~0.5 in observability study)

    def get_topology_checksum(self) -> int:
        """T087: Get checksum of current topology for divergence detection.

        Returns a hash/checksum of the current col_indices tensor that can be
        logged and compared across ranks to detect topology divergence.

        Returns:
            Integer checksum derived from col_indices

        Contract:
            - Read-only (doesn't modify state)
            - Deterministic: same col_indices produces same checksum
            - Fast: O(R*K) computation
        """
        # Use sum of indices multiplied by position for simple checksum
        # This catches both value changes and reordering
        with torch.no_grad():
            # Flatten and compute weighted sum
            flat = self.col_indices.flatten().long()
            positions = torch.arange(len(flat), device=flat.device, dtype=torch.long)
            # Use modular arithmetic to keep checksum in reasonable range
            weighted_sum = ((flat + 1) * (positions + 1)).sum().item()
            # Also include shape info
            checksum = int(weighted_sum) ^ (self.R << 20) ^ (self.K << 10)
            return checksum

    def topology_step(
        self,
        generator: Optional[torch.Generator] = None,
        save_snapshot: bool = True,
        global_step: Optional[int] = None,
        external_current_scores: Optional[Tensor] = None,
        external_candidate_scores: Optional[Tensor] = None,
    ) -> TopologyDecisionResult:
        """T033: Level 2 update: make topology decisions (prune/grow blocks).

        Call every ~100 training steps. Actions:
        1. T083: Sync scores across DDP ranks (if distributed)
        2. Score existing blocks by gradient EMA
        3. Score candidates by activation x error product
        4. Apply epsilon-greedy exploration
        5. Select top-K per row
        6. Swap low-scoring blocks for high-scoring candidates
        7. Initialize new block weights
        8. Reset all accumulators

        Args:
            generator: RNG for exploration. If None and global_step is provided,
                a deterministic generator is created for DDP consistency.
            save_snapshot: If True, save before/after col_indices to history (T077)
            global_step: T084: If provided, creates deterministic RNG with seed
                42 + global_step for consistent topology decisions across DDP ranks.
            external_current_scores: Optional [R, K] scores to override gradient EMA
                for existing blocks. Used by Experiment D to test alternative scorers
                (e.g., TileJEPA ranking, random). When None, uses self.block_score_ema.
            external_candidate_scores: Optional [R, C] scores to override activation×error
                candidate scores. When None, uses standard outer product.

        Returns:
            TopologyDecisionResult with swap statistics

        Contract:
            - Maintains exactly K active blocks per row
            - New blocks initialized with Kaiming x0.1 scale
            - Resets block_age to 0 for new blocks
            - All accumulators reset after decision
            - T083: Scores synced across ranks before decision in DDP
            - T084: Deterministic topology when global_step provided
        """
        import torch.distributed as dist

        # T055 (FR-030): Check if topology updates are disabled
        # This enables baseline comparison experiments with frozen random topology.
        # Still increment step counter for correct consolidation timing if later enabled.
        self._topology_step_count += 1

        if self.disable_topology_updates:
            # Return early with no changes
            return TopologyDecisionResult(
                num_swaps=0,
                swap_rate=0.0,
                pruned_positions=[],
                grown_columns=[],
                consolidation_triggered=False,
                crystallized_count=0,
                pressure_ratio=1.0,
            )

        # Topology warm-up: accumulate scores but don't swap during initial phase.
        # Gives gradient scoring time to build up meaningful signal before any
        # topology decisions are made.
        if self.topology_warmup_steps > 0 and global_step is not None and global_step < self.topology_warmup_steps:
            self._last_pruned_positions = []
            return TopologyDecisionResult(
                num_swaps=0,
                swap_rate=0.0,
                pruned_positions=[],
                grown_columns=[],
                consolidation_triggered=False,
                crystallized_count=0,
                pressure_ratio=1.0,
            )

        # T083: Sync scores across DDP ranks before making decisions
        if dist.is_initialized():
            self.sync_topology_scores()

        # T031: Compute pressure ratio for early consolidation detection
        pressure_ratio = self._compute_pressure_ratio()

        # T030/T031: Check consolidation triggers
        # - Regular interval: every consolidation_interval steps
        # - Early consolidation: pressure_ratio > 2.0 (distribution shift protection)
        consolidation_triggered = False
        crystallized_count = 0

        # Check regular interval trigger (FR-020)
        interval_trigger = (self._topology_step_count % self.consolidation_interval) == 0

        # Check early consolidation trigger (FR-023)
        # Only trigger if pressure > 2.0 AND we haven't already triggered via interval
        early_trigger = (pressure_ratio > 2.0) and (not interval_trigger)

        if interval_trigger or early_trigger:
            consolidation_result = self.consolidation_step()
            consolidation_triggered = True
            crystallized_count = consolidation_result["crystallized_count"]

        # T084: Create deterministic generator if global_step provided
        if generator is None and global_step is not None:
            generator = torch.Generator(device=self.col_indices.device)
            generator.manual_seed(42 + global_step)

        # T077: Save before topology snapshot
        before_indices = self.col_indices.clone() if save_snapshot else None

        # T043 (FR-019, FR-019b): Compute adaptive exploration epsilon based on pressure
        # Skip adaptive epsilon for first 100 topology steps (bootstrap period)
        if self._topology_step_count < 100:
            adjusted_epsilon = self.exploration_epsilon
        else:
            # Adjust epsilon based on pressure thresholds
            if pressure_ratio > self.pressure_high_threshold:  # > 1.5
                # High pressure: potential task boundary, increase exploration
                adjusted_epsilon = self.exploration_epsilon * 2.0
            elif pressure_ratio < self.pressure_low_threshold:  # < 0.7
                # Low pressure: dying gradients, boost exploration to escape local minimum
                adjusted_epsilon = self.exploration_epsilon * 1.5
            else:
                # Normal pressure: use base epsilon
                adjusted_epsilon = self.exploration_epsilon

        # Adaptive swap threshold: scale by inverse pressure ratio
        # High pressure (>1) = gradients changing = lower threshold = more swaps
        # Low pressure (<1) = stable = higher threshold = fewer swaps
        # Clamped to [0.5x, 3x] of base threshold to avoid extremes
        adaptive_threshold = self.swap_threshold / max(pressure_ratio, 0.01)
        adaptive_threshold = max(self.swap_threshold * 0.5, min(adaptive_threshold, self.swap_threshold * 3.0))

        # T044: Pass adjusted epsilon to TopologyScorer
        scorer = TopologyScorer(
            R=self.R,
            C=self.C,
            K=self.K,
            ema_alpha=self.score_ema_alpha,
            exploration_epsilon=adjusted_epsilon,
            swap_threshold=adaptive_threshold,
        )

        # Compute candidate scores (or use external override)
        if external_candidate_scores is not None:
            candidate_scores = external_candidate_scores
        else:
            candidate_scores = scorer.compute_candidate_scores(
                self.activation_norm_acc, self.error_norm_acc
            )

        # T021: Compute protection scores and apply to current block scores
        # Protection scores boost current_scores, making established blocks harder to replace.
        # Formula: protected_scores = current_scores * (1 + protection)
        # Where protection is in [0, 1], so boost ranges from 1x (no protection) to 2x (max protection)
        # Combined with swap_threshold (2.5x), fully protected blocks effectively require 5x better candidates.
        with torch.no_grad():
            protection_scores = self._compute_protection_scores()  # [R, K], values in [0, 1]
            base_current_scores = external_current_scores if external_current_scores is not None else self.block_score_ema

            # Composite scoring: weight by stability (applied at decision time, not accumulation)
            # Blocks with consistent gradients get higher effective scores than noisy ones
            stability = self._compute_stability()  # [R, K]
            composite_scores = base_current_scores * stability

            protected_current_scores = composite_scores * (1.0 + protection_scores)

            # === Fix #5: Churn Protection ===

            # Part A: Swap Cooldown - blocks in cooldown period get massive score boost
            # This makes them effectively un-swappable until cooldown expires
            effective_global_step = global_step if global_step is not None else self._topology_step_count * 100
            steps_since_swap = effective_global_step - self.last_swap_step
            cooldown_mask = steps_since_swap < self.swap_cooldown_steps  # [R, K]

            # Apply cooldown: boost scores of blocks in cooldown by 100x (effectively infinite protection)
            cooldown_boost = torch.where(
                cooldown_mask,
                torch.full_like(protected_current_scores, 100.0),
                torch.ones_like(protected_current_scores)
            )
            protected_current_scores = protected_current_scores * cooldown_boost

            # Part B: Age-based protection scaling (exponential curve)
            # Young blocks (recently swapped) get protection that decays as they mature
            # Protection = 1 - exp(-age / halflife), where halflife = swap_cooldown_steps / 5
            protection_halflife = max(1, self.swap_cooldown_steps // 5)
            age_protection = 1.0 - torch.exp(-self.block_age.float() / protection_halflife)
            # Scale current scores by age protection (young blocks get lower effective scores = harder to beat)
            # Wait, we want young blocks to be PROTECTED (not swapped), so we boost their scores
            young_block_boost = 1.0 + (1.0 - age_protection)  # Young blocks get up to 2x boost
            protected_current_scores = protected_current_scores * young_block_boost

            # Part E: Churn penalty - blocks swapped many times get penalized
            # Penalty = 1 / (1 + churn_penalty_factor * swap_count)
            # This REDUCES the effective score of "churny" positions, making them MORE likely to be swapped
            # Wait, that's backwards - we want to REDUCE churn, so churny positions should be PROTECTED
            # Actually the spec says "penalize churny blocks" but the goal is to reduce churn...
            # Let's interpret this as: reduce the CANDIDATE scores for churny positions
            # But candidate_scores is [R, C] not [R, K], and swap_count is per-slot not per-column
            # Better interpretation: churny positions have shown instability, so we boost their protection
            churn_protection = 1.0 + self.churn_penalty_factor * self.swap_count.float()
            protected_current_scores = protected_current_scores * churn_protection

        # T038 (FR-010, FR-011): Compute reserved columns for capacity reservation
        # Reserved columns get bonus scoring to encourage new tasks to use underutilized
        # capacity rather than fighting for established pathways.
        reserved_mask = self._compute_reserved_columns()

        # Select new topology (Fix 5: pass block_ages for age protection, if enabled)
        # Note: block_ages is now also factored into protection_scores via age term,
        # but we keep the separate age protection for backward compatibility
        # T022/T023: Pass swap cap parameters for per-row cap and per-layer budget
        # T037/T038: Pass reserved_mask and reserve_weight for capacity reservation
        new_col_indices, pruned_positions, grown_columns = scorer.select_top_k(
            current_scores=protected_current_scores,
            candidate_scores=candidate_scores,
            col_indices=self.col_indices,
            generator=generator,
            block_ages=self.block_age if self.use_age_protection else None,
            row_swap_cap_fraction=self.row_swap_cap_fraction,
            layer_swap_budget_fraction=self.layer_swap_budget_fraction,
            reserved_mask=reserved_mask,
            reserve_weight=self.reserve_weight,
        )

        # Store pruned positions for optimizer state reset (Fix 1)
        self._last_pruned_positions = list(pruned_positions)

        # Initialize new block weights (T034) - Fix #5: pass global_step for swap tracking
        effective_global_step = global_step if global_step is not None else self._topology_step_count * 100
        self._initialize_new_blocks(grown_columns, pruned_positions, new_col_indices, effective_global_step, candidate_scores=candidate_scores)

        # Update topology
        with torch.no_grad():
            self.col_indices.copy_(new_col_indices.to(torch.int32))

            # T010 (FR-003): Increment column usage counts from current col_indices
            # This tracks which columns are actively connected after topology changes
            self.col_usage_count.scatter_add_(
                0,
                self.col_indices.flatten().long(),
                torch.ones(self.R * self.K, device=self.col_indices.device, dtype=torch.float32)
            )

        # Invalidate column index cache since topology changed
        self.invalidate_column_index_cache()

        # Snapshot scores BEFORE reset for visualization/analysis
        with torch.no_grad():
            self._score_snapshot.copy_(self.block_score_ema)

        # Fix 3: Selective score reset - only reset scores for NEW blocks (swapped positions)
        # Surviving blocks keep their scores for continuity
        with torch.no_grad():
            if pruned_positions:
                # Create mask for positions that were swapped
                swapped_mask = torch.zeros_like(self.block_score_ema, dtype=torch.bool)
                for row, slot in pruned_positions:
                    swapped_mask[row, slot] = True

                # Only reset scores for swapped positions, preserve surviving block scores
                self.block_score_ema = torch.where(
                    swapped_mask,
                    torch.zeros_like(self.block_score_ema),
                    self.block_score_ema,
                )
            # Note: We no longer call self.block_score_ema.zero_() unconditionally

            # Reset activation/error accumulators (these should still be reset)
            self.activation_norm_acc.zero_()
            self.error_norm_acc.zero_()
        self._acc_steps = 0

        # T035: Compute result
        num_swaps = len(pruned_positions)
        total_blocks = self.R * self.K
        swap_rate = num_swaps / total_blocks if total_blocks > 0 else 0.0

        # T074: Track swap rate history
        self._swap_rate_history.append(swap_rate)
        if len(self._swap_rate_history) > self._swap_rate_history_max_size:
            self._swap_rate_history.pop(0)

        # T077: Save after topology snapshot
        if save_snapshot and before_indices is not None:
            self._topology_history.append((before_indices, self.col_indices.clone()))
            if len(self._topology_history) > self._topology_history_max_size:
                self._topology_history.pop(0)

        # T053 (FR-025 to FR-028): Emit telemetry event if enabled
        # Uses module-level telemetry writer from hooks module
        try:
            from titans_core.telemetry.hooks import emit_topology_metrics
            total_crystallized = int(self.crystallized_mask.sum().item())
            emit_topology_metrics(
                layer_name=getattr(self, '_layer_name', 'unknown'),
                swap_count=num_swaps,
                pressure_ratio=pressure_ratio,
                coherence_mean=0.0,  # Removed: coherence proved useless (~0.5 always)
                crystallized_count=total_crystallized,
            )
        except Exception:
            pass  # Never crash training for telemetry

        return TopologyDecisionResult(
            num_swaps=num_swaps,
            swap_rate=swap_rate,
            pruned_positions=pruned_positions,
            grown_columns=grown_columns,
            consolidation_triggered=consolidation_triggered,
            crystallized_count=crystallized_count,
            pressure_ratio=pressure_ratio,
        )

    def _initialize_new_blocks(
        self,
        grown_columns: List[int],
        pruned_positions: List[Tuple[int, int]],
        new_col_indices: Tensor,
        global_step: int = 0,
        candidate_scores: Optional[Tensor] = None,
    ) -> None:
        """T034/T059b/Fix5/WarmStart: Initialize weights and update buffers for newly added blocks.

        Replaces random Kaiming init with warm-start strategy to reduce convergence lag:
        - Strategy B (preferred): copy weights from same-column donor block (70/30 mix)
        - Strategy A (fallback): average same-row neighbor blocks (50/50 mix with noise)
        - Buffer updates are vectorized for speed

        Fix #5 Changes:
        - Updates last_swap_step to track cooldown
        - Increments swap_count for churn tracking
        - Preserves score history with decay instead of clearing

        Args:
            grown_columns: List of new column indices added (for reference)
            pruned_positions: List of (row, slot) that were pruned and need reinitialization
            new_col_indices: The new column indices tensor [R, K]
            global_step: Current training step for cooldown tracking
            candidate_scores: Optional [R, C] scores for new block positions (score-aware init)

        Buffer updates per position [row, slot]:
            - values[row, slot]: warm-start from neighbors (WarmStart)
            - block_age[row, slot] = 0 (original T034)
            - last_swap_step[row, slot] = global_step (Fix #5 Part A)
            - swap_count[row, slot] += 1 (Fix #5 Part E)
            - crystallized_mask[row, slot] = False (new blocks not crystallized)
            - score_history[row, slot, :]: decay, not clear (Fix #5 Part D)
        """
        if not pruned_positions:
            return

        # Kaiming initialization parameters (used as noise component in warm-start)
        B = self.tile_size
        fan_in = self.K * B
        gain = nn.init.calculate_gain("relu")
        std = gain / (fan_in**0.5) * 0.1  # Scale by 0.1

        # Compute layer-wide score statistics for score-aware initialization
        layer_median_score = self._compute_layer_median_score()

        # Vectorized indices for buffer updates
        rows = torch.tensor([r for r, s in pruned_positions], device=self.values.device)
        slots = torch.tensor([s for r, s in pruned_positions], device=self.values.device)

        with torch.no_grad():
            # Snapshot values BEFORE the loop to prevent copying from already-overwritten blocks.
            # Without this, if blocks A and B are both pruned and B tries to warm-start from A,
            # it would copy A's new (warm-started) weight instead of the original surviving weight.
            values_snapshot = self.values.data.clone()

            # Build set of pruned positions for exclusion from donor search
            pruned_set = set(pruned_positions)

            # --- Warm-start weight initialization ---
            for row, slot in pruned_positions:
                new_col = new_col_indices[row, slot].item()

                # Strategy B: same-column donor (most informative — same input feature)
                # Search for a SURVIVING block (not also being pruned) that maps to same column
                matches = (new_col_indices == new_col)
                matches[row] = False  # Exclude current row
                donor_found = False
                if matches.any():
                    for donor_pos in matches.nonzero(as_tuple=False).tolist():
                        donor_row, donor_slot = donor_pos
                        if (donor_row, donor_slot) not in pruned_set:
                            # Use snapshot to get original weights
                            donor_weight = values_snapshot[donor_row, donor_slot]
                            noise = torch.randn(B, B, device=self.values.device, dtype=self.values.dtype) * std * 0.3
                            self.values[row, slot] = 0.7 * donor_weight + 0.3 * noise
                            donor_found = True
                            break

                if not donor_found:
                    # Strategy A: same-row average (same output neurons, different input feature)
                    # Use snapshot for neighbor weights to avoid reading overwritten values
                    mask = torch.ones(self.K, dtype=torch.bool, device=self.values.device)
                    mask[slot] = False
                    # Also exclude other pruned slots in this row
                    for pr, ps in pruned_positions:
                        if pr == row and ps != slot:
                            mask[ps] = False
                    if mask.any():
                        neighbor_mean = values_snapshot[row, mask].mean(dim=0)
                        noise = torch.randn(B, B, device=self.values.device, dtype=self.values.dtype) * std
                        self.values[row, slot] = 0.5 * neighbor_mean + 0.5 * noise
                    else:
                        # Edge case: K=1 or all slots pruned — fall back to pure Kaiming
                        nn.init.normal_(self.values[row, slot], mean=0.0, std=std)

            # --- Vectorized buffer updates ---

            # Reset age for all swapped blocks
            self.block_age[rows, slots] = 0

            # Part A: Track when these blocks were swapped for cooldown
            self.last_swap_step[rows, slots] = global_step

            # Part E: Increment swap count for churn tracking
            self.swap_count[rows, slots] += 1

            # Clear crystallized status - new blocks must earn crystallization
            self.crystallized_mask[rows, slots] = False

            # Part D: Decay score history with score-aware initialization
            for i, (row, slot) in enumerate(pruned_positions):
                # Use candidate score for this block if available, else layer median
                if candidate_scores is not None:
                    new_col = new_col_indices[row, slot].item()
                    initial_score = candidate_scores[row, new_col].item()
                else:
                    initial_score = layer_median_score

                old_history = self.score_history[row, slot, :].clone()
                decayed = old_history * self.score_history_decay_on_swap
                zero_mask = decayed.abs() < 1e-8
                decayed[zero_mask] = initial_score
                self.score_history[row, slot, :] = decayed

    def _compute_layer_median_score(self) -> float:
        """Compute median non-zero score for initializing new blocks."""
        score_history_flat = self.score_history.view(-1, 10)
        nonzero_mask = score_history_flat.abs().sum(dim=1) > 0
        if nonzero_mask.any():
            return score_history_flat[nonzero_mask].median().item()
        return 0.0

    def get_topology_stats(self) -> TopologyStats:
        """Get current topology statistics for logging.

        Returns:
            TopologyStats with density, scores, ages, entropy, coherence, crystallization

        Contract:
            - Read-only (doesn't modify state)
            - Safe to call at any time
        """
        scorer = TopologyScorer(self.R, self.C, self.K)

        # Fix #5: Compute churn statistics
        # Note: blocks_in_cooldown requires knowing current step, estimate using topology_step_count
        estimated_current_step = self._topology_step_count * 100  # Estimate global step
        steps_since_swap = estimated_current_step - self.last_swap_step
        blocks_in_cooldown = int((steps_since_swap < self.swap_cooldown_steps).sum().item())

        return TopologyStats(
            density=self.K / self.C,
            avg_block_score=self.block_score_ema.mean().item(),
            avg_block_age=self.block_age.float().mean().item(),
            column_entropy=scorer.compute_column_entropy(self.col_indices),
            num_blocks=self.R * self.K,
            # mean_coherence kept at 0.0 for backward compat (coherence removed)
            mean_coherence=0.0,
            # T047: Count of crystallized (protected) blocks
            crystallized_count=int(self.crystallized_mask.sum().item()),
            # Fix #5: Churn statistics
            avg_swap_count=self.swap_count.float().mean().item(),
            blocks_in_cooldown=blocks_in_cooldown,
            max_swap_count=int(self.swap_count.max().item()),
        )

    def get_block_age_distribution(self) -> Dict[int, int]:
        """Get histogram of block ages.

        Returns distribution of block ages across all active blocks.
        Useful for monitoring topology diversity and turnover.

        Returns:
            Dict mapping age (int) to count (int)

        Contract:
            - Read-only (doesn't modify state)
            - Safe to call at any time
            - Ages are in topology steps (Level 2 updates)
        """
        # Get block_age tensor [R, K] and compute histogram
        ages = self.block_age.flatten().tolist()

        # Build histogram as dict
        histogram: Dict[int, int] = {}
        for age in ages:
            age_int = int(age)
            histogram[age_int] = histogram.get(age_int, 0) + 1

        return histogram

    def get_avg_swap_rate(self) -> float:
        """Get rolling average of swap rates over recent topology steps.

        Returns:
            Average swap rate (0.0 to 1.0), or 0.0 if no history

        Contract:
            - Read-only (doesn't modify state)
            - Uses rolling window of last 100 topology steps
        """
        if not self._swap_rate_history:
            return 0.0
        return sum(self._swap_rate_history) / len(self._swap_rate_history)

    def get_churn_statistics(self, current_step: Optional[int] = None) -> Dict[str, Any]:
        """Fix #5: Get detailed churn statistics for monitoring.

        Returns comprehensive statistics about block churn patterns,
        useful for diagnosing topology stability and tuning churn protection.

        Args:
            current_step: Current training step. If None, estimates from topology_step_count.

        Returns:
            Dict with keys:
                - avg_swap_count: Mean swap count across all blocks
                - max_swap_count: Maximum swap count (most churny position)
                - min_swap_count: Minimum swap count
                - swap_count_std: Standard deviation of swap counts
                - blocks_in_cooldown: Number of blocks currently in cooldown
                - cooldown_fraction: Fraction of blocks in cooldown
                - avg_steps_since_swap: Mean steps since last swap
                - recently_swapped_count: Blocks swapped in last 100 steps
                - swap_count_distribution: Dict of swap_count -> num_blocks
                - churn_hotspots: List of (row, slot) with highest swap counts

        Contract:
            - Read-only (doesn't modify state)
            - Safe to call at any time
        """
        with torch.no_grad():
            if current_step is None:
                current_step = self._topology_step_count * 100

            swap_counts = self.swap_count.float()
            steps_since_swap = current_step - self.last_swap_step.float()

            # Basic statistics
            avg_swap = swap_counts.mean().item()
            max_swap = int(swap_counts.max().item())
            min_swap = int(swap_counts.min().item())
            std_swap = swap_counts.std().item()

            # Cooldown statistics
            in_cooldown = (steps_since_swap < self.swap_cooldown_steps).sum().item()
            total_blocks = self.R * self.K
            cooldown_fraction = in_cooldown / total_blocks if total_blocks > 0 else 0.0

            # Recent activity
            avg_steps_since = steps_since_swap.mean().item()
            recently_swapped = (steps_since_swap < 100).sum().item()

            # Distribution
            swap_counts_flat = self.swap_count.flatten().tolist()
            distribution: Dict[int, int] = {}
            for count in swap_counts_flat:
                count = int(count)
                distribution[count] = distribution.get(count, 0) + 1

            # Find hotspots (positions with highest swap counts)
            top_k = min(10, total_blocks)
            flat_counts = self.swap_count.flatten()
            topk_values, topk_indices = torch.topk(flat_counts, top_k)
            hotspots = []
            for idx, count in zip(topk_indices.tolist(), topk_values.tolist()):
                row = idx // self.K
                slot = idx % self.K
                hotspots.append((row, slot, int(count)))

            return {
                "avg_swap_count": avg_swap,
                "max_swap_count": max_swap,
                "min_swap_count": min_swap,
                "swap_count_std": std_swap,
                "blocks_in_cooldown": int(in_cooldown),
                "cooldown_fraction": cooldown_fraction,
                "avg_steps_since_swap": avg_steps_since,
                "recently_swapped_count": int(recently_swapped),
                "swap_count_distribution": distribution,
                "churn_hotspots": hotspots,
            }

    def get_topology_history(self) -> List[Tuple[Tensor, Tensor]]:
        """Get saved topology snapshots.

        Returns:
            List of (before_col_indices, after_col_indices) tuples
            from recent topology_step() calls

        Contract:
            - Returns copies of the history (read-only)
            - Limited to last 10 snapshots
        """
        return list(self._topology_history)

    def get_density(self) -> float:
        """Get actual current density.

        Returns:
            K / C (should match configured density)
        """
        return self.K / self.C

    def zero_tile(self, r: int, k: int) -> None:
        """Zero a single tile at position (r, k). Works in both dense and sparse mode."""
        B = self.tile_size
        with torch.no_grad():
            if self._dense_mode:
                self.weight.data[r * B:(r + 1) * B, k * B:(k + 1) * B] = 0.0
            else:
                self.values.data[r, k] = 0.0

    def zero_tiles(self, mask: Tensor) -> None:
        """Zero all tiles where mask[r, k] is True. Works in both dense and sparse mode.

        Args:
            mask: [R, K] boolean tensor (True = zero this tile)
        """
        B = self.tile_size
        with torch.no_grad():
            if self._dense_mode:
                mask_expanded = mask.unsqueeze(2).unsqueeze(3).expand(-1, -1, B, B)
                weight_tiles = self.weight.data.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)
                weight_tiles[mask_expanded] = 0.0
            else:
                self.values.data[mask] = 0.0

    def get_swapped_positions(self) -> List[Tuple[int, int]]:
        """Return positions that were swapped in the last topology_step.

        Used for optimizer state reset after topology changes. The returned
        positions correspond to blocks that were pruned and replaced with
        new blocks, so their optimizer momentum/variance should be reset.

        Returns:
            List of (row, slot) tuples that were pruned in the last topology_step

        Contract:
            - Read-only (doesn't modify state)
            - Returns empty list if no topology_step has been called
            - Positions are valid indices into values tensor [R, K, B, B]
        """
        return list(self._last_pruned_positions)

    def get_block_features(self, row: int, slot: int) -> Tensor:
        """Extract rich per-block features for JEPA training.

        Returns a 10-dimensional feature vector capturing block stability signals:

        Feature Index | Name           | Range  | Description
        --------------|----------------|--------|----------------------------------
        0             | score_norm     | [0,1]  | Gradient EMA, normalized per-layer
        1             | age_norm       | [0,1]  | min(age/1000, 1.0)
        2             | stability2     | [0,1]  | 1/(1+variance) from score history (dup, coherence removed)
        3             | crystallized   | {0,1}  | Block is crystallized (protected)
        4             | row_position   | [0,1]  | r / R (output position)
        5             | slot_position  | [0,1]  | k / K (slot within row)
        6             | col_position   | [0,1]  | col_idx / C (input position)
        7             | stability      | [0,1]  | 1/(1+variance) from score history
        8             | score_trend    | [-1,1] | (recent-historical)/(historical+eps)
        9             | reserve_status | {0,1}  | Connects to underutilized column

        Args:
            row: Block row index (0 to R-1)
            slot: Block slot index within row (0 to K-1)

        Returns:
            Tensor [10] with normalized feature values

        Contract:
            - Read-only (doesn't modify state)
            - All features normalized to reasonable ranges
            - Safe to call during training
        """
        with torch.no_grad():
            features = torch.zeros(10, device=self.values.device, dtype=torch.float32)

            # 0. Score normalized: gradient EMA normalized by layer max
            max_score = self.block_score_ema.max().clamp(min=1e-8)
            features[0] = self.block_score_ema[row, slot] / max_score

            # 1. Age normalized: min(age/1000, 1.0) - saturates at ~1000 steps
            features[1] = min(self.block_age[row, slot].item() / 1000.0, 1.0)

            # 2. Stability: 1/(1+variance) from score history (coherence removed, proved useless)
            variance_2 = self.score_history[row, slot, :].var(unbiased=False)
            features[2] = 1.0 / (1.0 + variance_2)

            # 3. Crystallized: binary flag indicating protected status
            features[3] = float(self.crystallized_mask[row, slot])

            # 4. Row position: normalized output position
            features[4] = row / max(self.R - 1, 1)

            # 5. Slot position: normalized position within row
            features[5] = slot / max(self.K - 1, 1)

            # 6. Column position: actual input column position
            col_idx = self.col_indices[row, slot].item()
            features[6] = col_idx / max(self.C - 1, 1)

            # 7. Stability: 1/(1+variance) from score history ring buffer
            variance = self.score_history[row, slot, :].var(unbiased=False)
            features[7] = 1.0 / (1.0 + variance)

            # 8. Score trend: (recent - historical) / (historical + eps)
            recent_score = self.block_score_ema[row, slot]
            historical_score = self.block_score_historical_ema[row, slot]
            if historical_score.abs() > 1e-8:
                trend = (recent_score - historical_score) / (historical_score.abs() + 1e-8)
                features[8] = trend.clamp(-1.0, 1.0)
            else:
                features[8] = 0.0  # No historical data yet

            # 9. Reserve status: whether block connects to underutilized column
            mean_usage = self.col_usage_count.mean()
            threshold = self.reserve_threshold * mean_usage
            features[9] = float(self.col_usage_count[col_idx] < threshold)

            return features

    def get_all_block_features(self) -> Tensor:
        """Extract features for all blocks in the layer.

        Returns features for all R*K blocks, useful for batch processing
        or analysis. See get_block_features() for feature descriptions.

        Returns:
            Tensor [R, K, 10] with per-block feature vectors

        Contract:
            - Read-only (doesn't modify state)
            - More efficient than calling get_block_features() R*K times
        """
        with torch.no_grad():
            features = torch.zeros(
                self.R, self.K, 10, device=self.values.device, dtype=torch.float32
            )

            # 0. Score normalized: gradient EMA normalized by layer max
            max_score = self.block_score_ema.max().clamp(min=1e-8)
            features[:, :, 0] = self.block_score_ema / max_score

            # 1. Age normalized: min(age/1000, 1.0)
            features[:, :, 1] = (self.block_age.float() / 1000.0).clamp(max=1.0)

            # 2. Stability (coherence removed, proved useless ~0.5): 1/(1+variance)
            variance_2 = self.score_history.var(dim=-1, unbiased=False)  # [R, K]
            features[:, :, 2] = 1.0 / (1.0 + variance_2)

            # 3. Crystallized: binary flag
            features[:, :, 3] = self.crystallized_mask.float()

            # 4. Row position: normalized output position
            row_indices = torch.arange(self.R, device=self.values.device).float()
            features[:, :, 4] = (row_indices / max(self.R - 1, 1)).unsqueeze(1).expand(
                -1, self.K
            )

            # 5. Slot position: normalized position within row
            slot_indices = torch.arange(self.K, device=self.values.device).float()
            features[:, :, 5] = (slot_indices / max(self.K - 1, 1)).unsqueeze(0).expand(
                self.R, -1
            )

            # 6. Column position: actual input column position
            features[:, :, 6] = self.col_indices.float() / max(self.C - 1, 1)

            # 7. Stability: 1/(1+variance) from score history ring buffer
            variance = self.score_history.var(dim=-1, unbiased=False)  # [R, K]
            features[:, :, 7] = 1.0 / (1.0 + variance)

            # 8. Score trend: (recent - historical) / (historical + eps)
            historical_abs = self.block_score_historical_ema.abs() + 1e-8
            trend = (self.block_score_ema - self.block_score_historical_ema) / historical_abs
            features[:, :, 8] = trend.clamp(-1.0, 1.0)

            # 9. Reserve status: whether block connects to underutilized column
            mean_usage = self.col_usage_count.mean()
            threshold = self.reserve_threshold * mean_usage
            reserved_mask = self.col_usage_count < threshold  # [C]
            # Gather reserve status for each block's column
            col_indices_long = self.col_indices.long()  # [R, K]
            features[:, :, 9] = reserved_mask[col_indices_long].float()

            return features

    def reset_optimizer_state_for_swaps(self, optimizer: torch.optim.Optimizer) -> None:
        """Reset optimizer momentum/variance for recently swapped block positions.

        After topology_step() swaps blocks, the optimizer state (exp_avg, exp_avg_sq
        for Adam/AdamW) still contains momentum from the old block. This can cause
        training instability because the new block starts with stale gradients.

        This method zeroes the optimizer state for all positions that were swapped
        in the last topology_step().

        Args:
            optimizer: The optimizer whose state should be updated. Must have
                      self.values in its state dict (typically base_optimizer).

        Contract:
            - Safe to call even if no swaps occurred (no-op)
            - Safe to call if optimizer has no state for values (no-op)
            - Only modifies state for positions in _last_pruned_positions
            - Works with Adam, AdamW, and similar optimizers using exp_avg/exp_avg_sq

        Example:
            >>> layer = CMSBlockLinear(...)
            >>> result = layer.topology_step(global_step=100)
            >>> layer.reset_optimizer_state_for_swaps(optimizer)
        """
        if not self._last_pruned_positions:
            return

        param = self.values
        if param not in optimizer.state:
            return

        state = optimizer.state[param]
        with torch.no_grad():
            for row, slot in self._last_pruned_positions:
                # Reset first moment (momentum) for Adam/AdamW
                if "exp_avg" in state:
                    state["exp_avg"][row, slot].zero_()
                # Reset second moment (variance) for Adam/AdamW
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"][row, slot].zero_()
                # Also reset max exp_avg_sq if present (for AMSGrad variant)
                if "max_exp_avg_sq" in state:
                    state["max_exp_avg_sq"][row, slot].zero_()
                # Reset step count if per-parameter (some optimizers track per-param steps)
                if "step" in state and isinstance(state["step"], torch.Tensor):
                    if state["step"].shape == param.shape[:2]:
                        state["step"][row, slot] = 0

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return state for checkpointing.

        Includes:
            - values: Weight parameters
            - col_indices: Topology
            - bias: Bias parameters (if present)
            - block_score_ema: Scoring state
            - block_age: Block ages
            - accumulators: Accumulated norms
            - _acc_steps: Accumulator step counter
            - _swap_rate_history: Rolling swap rate history
            - _score_history_idx: Score history ring buffer index
            - _topology_step_count: Topology step counter for consolidation
        """
        # Use parent's state_dict which handles parameters and buffers
        state = super().state_dict(*args, **kwargs)
        # Add extra non-tensor state
        state["_acc_steps"] = self._acc_steps
        state["_swap_rate_history"] = list(self._swap_rate_history)
        state["_score_history_idx"] = self._score_history_idx
        state["_topology_step_count"] = self._topology_step_count
        # Note: topology_history is intentionally not saved (can be large)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Load state from checkpoint.

        Restores full layer state including topology and scoring.

        Args:
            state_dict: State dictionary from state_dict()
            strict: If True, raise error on missing/unexpected keys
        """
        # Extract extra state before parent load (which may modify dict)
        acc_steps = state_dict.pop("_acc_steps", 0)
        swap_rate_history = state_dict.pop("_swap_rate_history", [])
        score_history_idx = state_dict.pop("_score_history_idx", 0)
        topology_step_count = state_dict.pop("_topology_step_count", 0)

        # Migration: remove buffers that were deleted (coherence tracking)
        # so old checkpoints don't fail with strict=True
        _removed_keys = ["gradient_coherence_ema", "prev_grad_direction"]
        for key in _removed_keys:
            state_dict.pop(key, None)

        # Migration: handle dtype changes (int64→int32, fp32→bf16) for buffers
        # Old checkpoints may have different dtypes; let PyTorch handle the cast
        # by loading with strict=False for known-changed buffers, then strict for the rest
        for key in ["last_swap_step", "swap_count", "score_history", "block_score_historical_ema"]:
            if key in state_dict:
                expected = getattr(self, key, None)
                if expected is not None and state_dict[key].dtype != expected.dtype:
                    state_dict[key] = state_dict[key].to(expected.dtype)

        # Load parameters and buffers
        super().load_state_dict(state_dict, strict=strict)

        # Restore extra state
        self._acc_steps = acc_steps
        self._swap_rate_history = list(swap_rate_history)
        self._score_history_idx = score_history_idx
        self._topology_step_count = topology_step_count
        # Reset topology history on load (not saved)
        self._topology_history = []

    def compact(self) -> int:
        """Transition from dense weight to Block-ELL sparse format.

        Pre-compact (dense mode): weight is [out, in], pruned tiles are zeroed regions.
        Post-compact (sparse mode): weight is Block-ELL [R, new_K, B, B] with col_indices.

        Strategy:
        - Reshape dense weight into tiles [R, C, B, B]
        - Detect active tiles (Frobenius norm > 0)
        - new_K = min active count across rows (at least 1)
        - Extract alive tiles into Block-ELL values + col_indices
        - Delete dense weight parameter, create sparse values parameter
        - Switch forward path from F.linear to Triton Block-ELL kernel

        Returns:
            new_K: The compacted number of active blocks per row.
        """
        with torch.no_grad():
            B = self.tile_size

            if self._dense_mode:
                # ── Dense → Sparse transition ──────────────────────────────
                device = self.weight.device
                dtype = self.weight.dtype

                # Apply final ternary quantization if active
                w_data = self.weight.data
                if self._ternary_mode:
                    w_data = self._ternary_ste(self.weight).detach()
                    self._ternary_mode = False

                # Reshape dense weight into tile view [R, C, B, B]
                all_tiles = w_data.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3).contiguous()

                # Detect active tiles
                block_norms = all_tiles.norm(dim=(-2, -1))  # [R, C]
                active_mask = block_norms > 0.0

                active_counts = active_mask.sum(dim=1)
                new_K = int(active_counts.min().clamp(min=1).item())

                # Score-ranked selection: active tiles get priority
                scores = self.block_score_ema.clone()
                scores = scores + active_mask.float() * 1e9
                _, top_indices = scores.topk(new_K, dim=1, largest=True, sorted=True)

                # Gather tiles into Block-ELL format
                idx_4d = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, B, B)
                new_values_data = all_tiles.gather(1, idx_4d.to(device))
                new_col_indices = top_indices.to(torch.int32)

                # Delete dense parameter, create sparse parameter
                del self._parameters["weight"]
                new_param = nn.Parameter(new_values_data.clone().to(dtype=dtype, device=device))
                self.values = new_param
                self._dense_mode = False
            else:
                # ── Sparse → Sparse re-compact ─────────────────────────────
                device = self.values.device
                dtype = self.values.dtype

                block_norms = self.values.data.norm(dim=(-2, -1))
                active_mask = block_norms > 0.0

                active_counts = active_mask.sum(dim=1)
                new_K = int(active_counts.min().clamp(min=1).item())

                if new_K >= self.K:
                    return self.K

                scores = self.block_score_ema.clone()
                scores = scores + active_mask.float() * 1e9
                _, top_indices = scores.topk(new_K, dim=1, largest=True, sorted=True)

                def _gather_rk(t: Tensor) -> Tensor:
                    if t.dim() == 2:
                        return t.gather(1, top_indices.to(t.device))
                    elif t.dim() == 3:
                        idx = top_indices.unsqueeze(-1).expand(-1, -1, t.size(2)).to(t.device)
                        return t.gather(1, idx)
                    elif t.dim() == 4:
                        idx = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t.size(2), t.size(3)).to(t.device)
                        return t.gather(1, idx)
                    else:
                        raise ValueError(f"compact: unexpected tensor shape {t.shape}")

                new_values_data = _gather_rk(self.values.data)
                new_col_indices = _gather_rk(self.col_indices).to(torch.int32)

                del self._parameters["values"]
                new_param = nn.Parameter(new_values_data.clone().to(dtype=dtype, device=device))
                self.values = new_param

            # ── Common: resize all [R, K, ...] buffers ─────────────────────
            def _gather_buf(t: Tensor) -> Tensor:
                if t.dim() == 2:
                    return t.gather(1, top_indices.to(t.device))
                elif t.dim() == 3:
                    idx = top_indices.unsqueeze(-1).expand(-1, -1, t.size(2)).to(t.device)
                    return t.gather(1, idx)
                return t

            def _replace_buf(name: str, new_tensor: Tensor) -> None:
                self.register_buffer(name, new_tensor.to(device=device))

            _replace_buf("col_indices", new_col_indices)
            _replace_buf("block_score_ema", _gather_buf(self.block_score_ema))
            _replace_buf("block_age", _gather_buf(self.block_age).to(torch.int32))
            _replace_buf("_score_snapshot", _gather_buf(self._score_snapshot))
            _replace_buf("score_history", _gather_buf(self.score_history))
            _replace_buf("crystallized_mask", _gather_buf(self.crystallized_mask.int()).bool())
            _replace_buf("block_score_historical_ema", _gather_buf(self.block_score_historical_ema))
            _replace_buf("last_swap_step", _gather_buf(self.last_swap_step).to(torch.int32))
            _replace_buf("swap_count", _gather_buf(self.swap_count).to(torch.int32))

            # ── Update metadata ────────────────────────────────────────────
            self.K = new_K
            self.density = new_K / self.C
            self._block_ell_config = BlockELLConfig(R=self.R, C=self.C, K=self.K, B=self.tile_size)
            self.disable_topology_updates = True
            self.invalidate_column_index_cache()
            self._last_pruned_positions = []

        return new_K

    def compact_with_groups(self, n_clusters: int = 16) -> int:
        """Compact and assign row-groups to clusters for routed forward pass.

        Calls compact() first to rebuild the Block-ELL layout with reduced K,
        then divides the R row-groups into n_clusters contiguous clusters.
        Registers cluster metadata buffers used by routed_forward().

        Args:
            n_clusters: Number of row-group clusters for routing.

        Returns:
            new_K: The compacted number of active blocks per row.
        """
        new_K = self.compact()

        # Assign rows to clusters (contiguous blocks, roughly equal size)
        rows_per_cluster = self.R // n_clusters
        remainder = self.R % n_clusters

        starts, ends = [], []
        row = 0
        for c in range(n_clusters):
            s = row
            row += rows_per_cluster + (1 if c < remainder else 0)
            starts.append(s)
            ends.append(row)

        self.n_clusters = n_clusters
        self.register_buffer(
            "cluster_starts", torch.tensor(starts, dtype=torch.int32)
        )
        self.register_buffer(
            "cluster_ends", torch.tensor(ends, dtype=torch.int32)
        )

        # Pre-compute output-dim to cluster mapping for fast indexing
        # output_to_cluster[i] = cluster index for output feature i
        output_to_cluster = torch.zeros(self.out_features, dtype=torch.long)
        for c in range(n_clusters):
            s = starts[c] * self.tile_size
            e = ends[c] * self.tile_size
            output_to_cluster[s:e] = c
        self.register_buffer("output_to_cluster", output_to_cluster)

        return new_K

    @torch.no_grad()
    def compact_with_reorder(
        self,
        fc_counterpart: "CMSBlockLinear",
        macro_tile_size: int = 32,
        is_fc1: bool = True,
    ) -> Tuple[int, int]:
        """Compact and reorder d_ff columns for maximum macro-block density.

        Calls compact() first to rebuild the Block-ELL layout with reduced K,
        then permutes the shared d_ff dimension so alive tiles cluster toward
        low column/row indices.  After permutation, tail macro-blocks are fully
        dead and the Triton kernel can skip them entirely.

        This method must be called on the fc1 layer (is_fc1=True) and passed
        the paired fc2 layer, OR called on fc2 (is_fc1=False) with fc1 as the
        counterpart.  Convention: always call on fc1 and pass fc2.

        Terminology
        -----------
        fc1 : d_model → d_ff   (this layer when is_fc1=True)
              self.R = d_ff / B,  self.C = d_model / B
              col_indices index into d_model block-columns
              BUT for the *column-reorder* view, fc1's OUTPUT rows are d_ff.
              The Block-ELL col_indices for fc1 point into d_model (input dim),
              while the OUTPUT of fc1 has shape [R_fc1, B] = [d_ff/B * B = d_ff].
              So fc1.R == C_ff and the alive ROW of fc1 corresponds to a d_ff column.

        Wait — let's be precise about CMSBlockLinear orientation:
          CMSBlockLinear(in_features, out_features)
          self.R = out_features // tile_size   (block-rows, output dim)
          self.C = in_features  // tile_size   (block-cols, input dim)
          col_indices [R, K] point into input block-columns (0..C-1)

        For fc1(d_model, d_ff):
          R_fc1 = d_ff / B   ← these rows ARE the d_ff tile-columns we permute
          C_fc1 = d_model / B
          col_indices → input (d_model)
          So reordering fc1 ROWS permutes the d_ff output → is_column_dim=False for fc1!

        For fc2(d_ff, d_model):
          R_fc2 = d_model / B
          C_fc2 = d_ff / B   ← col_indices point into d_ff tile-columns
          So remapping fc2 col_indices permutes the d_ff input → is_column_dim=True for fc2!

        In the JAX code the fc1/fc2 roles are named from the *caller's* perspective
        where fc1 col_indices index d_ff.  Here we derive alive masks from block norms
        and pass the correct is_column_dim flags.

        Args:
            fc_counterpart: The paired layer (fc2 if self is fc1, or vice versa).
            macro_tile_size: Number of tile-columns per macro-block.  Default 32
                             is the sweet spot for the RTX 5090 (SM_120).
            is_fc1: True if self is the fc1 (d_model→d_ff) layer.

        Returns:
            (new_K_self, k_active_macros) — compacted K for self, and the number
            of macro-block columns that contain at least one alive tile after reorder.
        """
        from titans_core.opt.column_reorder import (
            compute_column_importance,
            compute_permutation,
            apply_permutation_to_block_ell,
            compute_k_active_macros,
        )

        # ------------------------------------------------------------------
        # 1. Compact self (and optionally counterpart if not yet compacted)
        # ------------------------------------------------------------------
        new_K_self = self.compact()
        # Compact the counterpart too so both have consistent sparse layout
        fc_counterpart.compact()

        # ------------------------------------------------------------------
        # 2. Determine which layer is fc1 (d_model→d_ff) and which is fc2
        # ------------------------------------------------------------------
        if is_fc1:
            fc1_layer = self
            fc2_layer = fc_counterpart
        else:
            fc1_layer = fc_counterpart
            fc2_layer = self

        # C_ff = number of d_ff tile-columns = fc1.R = fc2.C
        C_ff = fc1_layer.R
        assert fc2_layer.C == C_ff, (
            f"compact_with_reorder: d_ff mismatch — fc1.R={C_ff} but fc2.C={fc2_layer.C}"
        )

        # ------------------------------------------------------------------
        # 3. Build alive masks from block Frobenius norms
        # ------------------------------------------------------------------
        fc1_alive = (fc1_layer.values.data.norm(dim=(-2, -1)) > 0.0)  # [R_ff, K_fc1]
        fc2_alive = (fc2_layer.values.data.norm(dim=(-2, -1)) > 0.0)  # [R_dmodel, K_fc2]

        # ------------------------------------------------------------------
        # 4. Compute column importance and permutation
        #    fc1 ROW dimension == d_ff (is_column_dim=False for fc1 values/rows)
        #    fc2 COL_INDICES    == d_ff (is_column_dim=True  for fc2 col_indices)
        #
        #    For importance scoring we treat:
        #      fc1_alive_mask = fc1_alive  [R_ff, K_fc1]  (rows = d_ff)
        #      fc2_alive_mask = fc2_alive  [R_dmodel, K_fc2]  (col_indices → d_ff)
        #    We need to feed compute_column_importance in JAX convention:
        #      fc1_alive_mask[r, k] = alive for fc1 tile at (d_ff row r, k-slot k)
        #      fc2_alive_mask used for per-row alive count
        #    But our fc2_alive is [R_dmodel, K_fc2] — rows are d_model, not d_ff.
        #    For fc2, the column importance comes from fc2.col_indices pointing into d_ff.
        #    We do this manually below instead of using compute_column_importance's
        #    fc2_row_counts path (which assumes R_ff == C_ff layout).
        # ------------------------------------------------------------------
        device = fc1_layer.values.device

        # fc1 contribution: fc1 rows ARE d_ff columns; count alive tiles per d_ff row
        fc1_row_counts = fc1_alive.float().sum(dim=1)  # [C_ff]

        # fc2 contribution: fc2.col_indices[r, k] indexes d_ff tile-column
        flat_fc2_cols = fc2_layer.col_indices.reshape(-1).to(torch.int64)   # [R_dmodel * K_fc2]
        flat_fc2_alive = fc2_alive.reshape(-1).float()                        # [R_dmodel * K_fc2]
        valid_fc2 = (flat_fc2_cols >= 0).float()
        safe_fc2_cols = flat_fc2_cols.clamp(0, C_ff - 1)
        fc2_col_counts = torch.zeros(C_ff, dtype=torch.float32, device=device)
        fc2_col_counts.scatter_add_(0, safe_fc2_cols, flat_fc2_alive * valid_fc2)

        importance = fc1_row_counts + fc2_col_counts  # [C_ff]
        perm = compute_permutation(importance)          # [C_ff] int32

        # ------------------------------------------------------------------
        # 5. Apply permutation to fc1 (row dim — physically reorder rows)
        # ------------------------------------------------------------------
        perm_long = perm.long()

        # fc1 rows are d_ff: reorder physically
        fc1_new_values = fc1_layer.values.data[perm_long]       # [C_ff, K_fc1, B, B]
        fc1_new_cols = fc1_layer.col_indices[perm_long]          # [C_ff, K_fc1]
        fc1_new_alive = fc1_alive[perm_long]                     # [C_ff, K_fc1]

        # fc1 also has [R, K] metadata buffers that must follow
        def _reorder_fc1_buf_rows(t: Tensor) -> Tensor:
            if t.dim() >= 1 and t.shape[0] == C_ff:
                return t[perm_long]
            return t

        fc1_layer.values.data.copy_(fc1_new_values)
        fc1_layer.col_indices.copy_(fc1_new_cols)
        for buf_name in [
            "block_score_ema", "block_age", "_score_snapshot", "score_history",
            "crystallized_mask", "block_score_historical_ema",
            "last_swap_step", "swap_count",
        ]:
            buf = getattr(fc1_layer, buf_name, None)
            if buf is not None and buf.shape[0] == C_ff:
                buf.copy_(_reorder_fc1_buf_rows(buf))

        # ------------------------------------------------------------------
        # 6. Apply permutation to fc2 (col_indices — remap d_ff references)
        # ------------------------------------------------------------------
        # Build inverse permutation: inv_perm[old_d_ff_col] = new_d_ff_col
        inv_perm = torch.zeros(C_ff, dtype=torch.int32, device=device)
        inv_perm[perm_long] = torch.arange(C_ff, dtype=torch.int32, device=device)

        sentinel_mask = fc2_layer.col_indices < 0
        safe_cols = fc2_layer.col_indices.clamp(0, C_ff - 1).long()
        new_fc2_col_indices = torch.where(
            sentinel_mask,
            torch.full_like(fc2_layer.col_indices, -1),
            inv_perm[safe_cols],
        )
        fc2_layer.col_indices.copy_(new_fc2_col_indices)

        # Invalidate column index caches (col_indices changed)
        fc1_layer.invalidate_column_index_cache()
        fc2_layer.invalidate_column_index_cache()

        # ------------------------------------------------------------------
        # 7. Compute and store k_active_macros on both layers
        # ------------------------------------------------------------------
        # Use fc1 new col_indices — they are d_model indices, not d_ff.
        # k_active_macros in d_ff space: use fc2.col_indices (which point into d_ff).
        fc2_alive_new = (fc2_layer.values.data.norm(dim=(-2, -1)) > 0.0)
        k_active = compute_k_active_macros(
            alive_mask=fc2_alive_new,
            col_indices=fc2_layer.col_indices,
            tiles_per_macro=macro_tile_size // fc2_layer.tile_size,
            C_ff=C_ff,
        )

        self.k_active_macros = k_active
        self.macro_tile_size = macro_tile_size
        fc_counterpart.k_active_macros = k_active
        fc_counterpart.macro_tile_size = macro_tile_size

        return new_K_self, k_active

    def routed_forward(
        self, x: Tensor, cluster_indices: Tensor, cluster_weights: Tensor
    ) -> Tensor:
        """Forward pass with per-token cluster routing (top-k style).

        Computes full Block-ELL forward, then scales each output feature
        by the weight of its cluster. Non-selected clusters get weight 0.
        Gradient flows through the continuous cluster_weights to the router.

        Requires compact_with_groups() to have been called first.

        Args:
            x: Input [B, T, in_features]
            cluster_indices: [B, T, top_k] active cluster indices per token
            cluster_weights: [B, T, top_k] softmax weights (pre-scaled by top_k)

        Returns:
            Routed output [B, T, out_features]
        """
        B, T, _ = x.shape

        # Full Block-ELL forward (all rows computed)
        raw_output = self.forward(x)  # [B, T, out_features]

        # Build per-cluster weight map: [B, T, n_clusters] with 0 for non-selected
        cluster_weight_full = torch.zeros(
            B, T, self.n_clusters, device=x.device, dtype=raw_output.dtype
        )
        cluster_weight_full.scatter_(
            -1, cluster_indices.long(), cluster_weights.to(raw_output.dtype)
        )

        # Expand to per-output-feature scaling via pre-computed mapping
        # output_to_cluster: [out_features] → cluster_id
        # cluster_weight_full[:, :, output_to_cluster] → [B, T, out_features]
        output_scale = cluster_weight_full[:, :, self.output_to_cluster]

        return raw_output * output_scale

    def gated_forward(self, x: Tensor, gates: Tensor) -> Tensor:
        """Forward pass with continuous ReLU gates (ReMoE style).

        Computes full Block-ELL forward, then scales each output feature
        by its cluster's gate value. Gate=0 (ReLU inactive) zeros the output.
        Fully differentiable — no discrete selection.

        Requires compact_with_groups() to have been called first.

        Args:
            x: Input [B, T, in_features]
            gates: [B, T, n_clusters] — continuous gate values from ReLU router.
                   0 for inactive clusters, positive for active.

        Returns:
            Gated output [B, T, out_features]
        """
        # Full Block-ELL forward (all rows computed)
        raw_output = self.forward(x)  # [B, T, out_features]

        # Expand gates to per-output-feature scaling
        # output_to_cluster: [out_features] → cluster_id
        output_scale = gates[:, :, self.output_to_cluster]  # [B, T, out_features]

        return raw_output * output_scale

    def routed_forward_v2(self, x: Tensor, k_active: Tensor) -> Tensor:
        """Forward pass with per-token variable K using the routed Triton kernel.

        Unlike routed_forward() / gated_forward() which compute all tiles then
        mask, this actually skips computation for inactive tile columns at the
        kernel level via per-token k_active masking.

        Requires compact_with_reorder() to have been called first so that alive
        tiles are packed toward low column/row indices (leftward), making the
        per-token k_active cutoff meaningful.

        Args:
            x: Input [B, T, in_features] or [batch, in_features].
            k_active: Per-token active block count [B, T] or [batch] int32.
                      Must satisfy 0 <= k_active[token] <= self.K (current K).

        Returns:
            Output [B, T, out_features] or [batch, out_features].
        """
        from titans_core.kernels.block_ell_routed_forward import (
            block_ell_routed_forward_autograd,
        )

        bias = self.bias if self.bias is not None else None
        return block_ell_routed_forward_autograd(
            x, self.values, self.col_indices, k_active, bias=bias
        )

    def to_dense(self) -> Tensor:
        """Convert current topology to dense weight matrix.

        Returns:
            Dense weight tensor [out_features, in_features]

        Use for:
            - Debugging / visualization
            - Comparison with nn.Linear
            - Export to frameworks without sparse support
        """
        from .block_ell import to_dense as block_ell_to_dense

        return block_ell_to_dense(
            values=self.values,
            col_indices=self.col_indices,
            R=self.R,
            C=self.C,
            K=self.K,
            B=self.tile_size,
        )

    @classmethod
    def from_dense(
        cls,
        dense_layer: nn.Linear,
        tile_size: int = 16,
        density: float = 0.5,
        score_ema_alpha: float = 0.95,
        swap_threshold: float = 1.5,
        exploration_epsilon: float = 0.05,
        use_age_protection: bool = True,
    ) -> "CMSBlockLinear":
        """Create sparse layer from existing dense layer.

        Initializes topology by magnitude-based pruning of dense weights.
        Uses Frobenius norm of each block to select the top-K most important
        blocks per row, preserving the highest magnitude weights.

        Args:
            dense_layer: Source nn.Linear layer
            tile_size: Block size for sparse format
            density: Target density (fraction of blocks to keep per row)
            score_ema_alpha: EMA momentum for gradient scores (default 0.95)
            swap_threshold: Required improvement ratio for topology swaps (default 1.5)
            exploration_epsilon: Random swap probability for diversity (default 0.05)
            use_age_protection: Apply age-based score bonus (default True)

        Returns:
            CMSBlockLinear initialized from dense weights

        Raises:
            ValueError: If dense layer dimensions not divisible by tile_size

        Example:
            >>> dense = nn.Linear(128, 256)
            >>> sparse = CMSBlockLinear.from_dense(dense, tile_size=16, density=0.5)
            >>> # sparse now has topology based on dense weight magnitudes
        """
        from .block_ell import from_dense as block_ell_from_dense

        in_features = dense_layer.in_features
        out_features = dense_layer.out_features
        has_bias = dense_layer.bias is not None

        # Validate dimensions
        if in_features % tile_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by tile_size ({tile_size})"
            )
        if out_features % tile_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by tile_size ({tile_size})"
            )

        # Convert dense weights to block-ELL format using magnitude-based selection
        dense_weight = dense_layer.weight.data  # [out_features, in_features]
        values, col_indices, R, C, K, B = block_ell_from_dense(
            dense_weight, tile_size=tile_size, density=density
        )

        # Create the sparse layer (this will initialize with random topology)
        sparse_layer = cls(
            in_features=in_features,
            out_features=out_features,
            tile_size=tile_size,
            density=density,
            bias=has_bias,
            score_ema_alpha=score_ema_alpha,
            swap_threshold=swap_threshold,
            exploration_epsilon=exploration_epsilon,
            use_age_protection=use_age_protection,
            device=dense_layer.weight.device,
            dtype=dense_layer.weight.dtype,
        )

        # Override with the values and topology from magnitude-based selection
        with torch.no_grad():
            sparse_layer.values.copy_(values)
            sparse_layer.col_indices.copy_(col_indices.to(torch.int32))

            # Copy bias if present
            if has_bias:
                sparse_layer.bias.copy_(dense_layer.bias.data)

        return sparse_layer

    def extra_repr(self) -> str:
        """String representation for print(layer)."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tile_size={self.tile_size}, density={self.density:.2f}, "
            f"K={self.K}, bias={self.bias is not None}"
        )
