"""Topology scoring utilities for CMS block-sparse layers.

TopologyScorer manages the gradient-based importance metrics used to make
topology decisions in CMSBlockLinear layers. It handles:
- Gradient EMA accumulation for existing blocks
- Candidate scoring using activation x error product
- Epsilon-greedy exploration for topology diversity
- Top-K selection per row
- Exploration epsilon decay for convergence (Fix #4)

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class ExplorationEpsilonDecay:
    """Configuration for exploration epsilon decay in TopologyScorer (Fix #4).

    Attributes:
        decay_type: Type of decay schedule ("linear", "exponential", "cosine")
        initial_epsilon: Starting exploration epsilon (default 0.05)
        final_epsilon: Minimum epsilon at end of training (default 0.001)
        decay_start_step: Step to start decay (default 10000)
        decay_half_life: For exponential decay, steps to halve epsilon
    """
    decay_type: str = "cosine"
    initial_epsilon: float = 0.05
    final_epsilon: float = 0.001
    decay_start_step: int = 10000
    decay_half_life: int = 50000


@dataclass
class BlockScore:
    """Per-block importance score with metadata.

    Attributes:
        row_idx: Output block-row index [0, R)
        slot_idx: Slot within row [0, K)
        col_idx: Input block-column index [0, C)
        gradient_ema: EMA of gradient Frobenius norm
        age: Steps since block creation
    """

    row_idx: int
    slot_idx: int
    col_idx: int
    gradient_ema: float
    age: int


@dataclass
class CandidateScore:
    """Score for a potential new block position.

    Attributes:
        row_idx: Output block-row index
        col_idx: Candidate input block-column
        score: activation_norm[col] * error_norm[row]
        is_active: Whether this position is currently active
    """

    row_idx: int
    col_idx: int
    score: float
    is_active: bool


class TopologyScorer:
    """Manages topology scoring and selection for block-sparse layers.

    This class implements the scoring logic for CMS Level 2 topology updates.
    It is designed to work with CMSBlockLinear but can be used standalone
    for testing and debugging.

    Args:
        R: Number of output block-rows
        C: Number of input block-columns
        K: Active blocks per row
        ema_alpha: Momentum for gradient EMA (default 0.95)
        exploration_epsilon: Random swap probability (default 0.05)
        swap_threshold: Required improvement ratio for swap (default 1.5)
        epsilon_decay_config: Optional config for exploration epsilon decay (Fix #4)

    Example:
        >>> scorer = TopologyScorer(R=160, C=40, K=20)
        >>> grad_norms = torch.randn(160, 20).abs()  # [R, K]
        >>> scorer.update_gradient_ema(grad_norms)

        # With exploration decay (Fix #4):
        >>> scorer.update_exploration_epsilon(step=50000, total_steps=100000)
    """

    def __init__(
        self,
        R: int,
        C: int,
        K: int,
        ema_alpha: float = 0.95,
        exploration_epsilon: float = 0.05,
        swap_threshold: float = 2.5,
        epsilon_decay_config: Optional[ExplorationEpsilonDecay] = None,
    ) -> None:
        """Initialize topology scorer."""
        if K > C:
            raise ValueError(f"K ({K}) cannot exceed C ({C})")
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha ({ema_alpha}) must be in [0, 1]")
        if not (0.0 <= exploration_epsilon <= 0.5):
            raise ValueError(f"exploration_epsilon ({exploration_epsilon}) must be in [0, 0.5]")

        self.R = R
        self.C = C
        self.K = K
        self.ema_alpha = ema_alpha
        self.exploration_epsilon = exploration_epsilon
        self._initial_epsilon = exploration_epsilon  # Store initial value
        self.swap_threshold = swap_threshold

        # Exploration decay config (Fix #4)
        self.epsilon_decay_config = epsilon_decay_config or ExplorationEpsilonDecay(
            initial_epsilon=exploration_epsilon
        )

    def update_exploration_epsilon(
        self,
        step: int,
        total_steps: int,
    ) -> float:
        """Update exploration epsilon based on training progress (Fix #4).

        The exploration epsilon decays over training to encourage convergence
        to a stable topology. This method updates `self.exploration_epsilon`
        and returns the new value.

        Args:
            step: Current training step
            total_steps: Total steps in training

        Returns:
            New exploration epsilon value
        """
        decay_cfg = self.epsilon_decay_config

        # Before decay starts, use initial epsilon
        if step < decay_cfg.decay_start_step:
            self.exploration_epsilon = decay_cfg.initial_epsilon
            return self.exploration_epsilon

        # Calculate progress through decay phase
        decay_steps = total_steps - decay_cfg.decay_start_step
        if decay_steps <= 0:
            self.exploration_epsilon = decay_cfg.final_epsilon
            return self.exploration_epsilon

        progress = min(1.0, (step - decay_cfg.decay_start_step) / decay_steps)

        if decay_cfg.decay_type == "linear":
            # Linear: epsilon = initial - progress * (initial - final)
            epsilon = decay_cfg.initial_epsilon - progress * (
                decay_cfg.initial_epsilon - decay_cfg.final_epsilon
            )

        elif decay_cfg.decay_type == "exponential":
            # Exponential: epsilon = initial * (0.5 ^ (steps / half_life))
            steps_since_decay = step - decay_cfg.decay_start_step
            epsilon = decay_cfg.initial_epsilon * (
                0.5 ** (steps_since_decay / decay_cfg.decay_half_life)
            )
            epsilon = max(epsilon, decay_cfg.final_epsilon)

        elif decay_cfg.decay_type == "cosine":
            # Cosine annealing: smooth decay from initial to final
            epsilon = decay_cfg.final_epsilon + (
                decay_cfg.initial_epsilon - decay_cfg.final_epsilon
            ) * 0.5 * (1 + math.cos(math.pi * progress))

        else:
            # Fallback to linear
            epsilon = decay_cfg.initial_epsilon - progress * (
                decay_cfg.initial_epsilon - decay_cfg.final_epsilon
            )

        self.exploration_epsilon = max(decay_cfg.final_epsilon, epsilon)
        return self.exploration_epsilon

    def get_exploration_stats(self) -> dict:
        """Get exploration epsilon statistics for logging (Fix #4).

        Returns:
            Dict with current epsilon, initial, final, and decay config
        """
        return {
            "exploration_epsilon": self.exploration_epsilon,
            "exploration_epsilon_initial": self.epsilon_decay_config.initial_epsilon,
            "exploration_epsilon_final": self.epsilon_decay_config.final_epsilon,
            "exploration_decay_type": self.epsilon_decay_config.decay_type,
        }

    def update_gradient_ema(
        self,
        grad_norms: Tensor,
        current_ema: Tensor,
    ) -> Tensor:
        """Update gradient EMA with new gradient norms.

        Implements: ema = alpha * new + (1 - alpha) * old

        Args:
            grad_norms: New gradient Frobenius norms [R, K]
            current_ema: Current EMA values [R, K]

        Returns:
            Updated EMA tensor [R, K]
        """
        # EMA formula: new_ema = alpha * grad_norms + (1 - alpha) * current_ema
        return self.ema_alpha * grad_norms + (1.0 - self.ema_alpha) * current_ema

    def compute_candidate_scores(
        self,
        activation_norms: Tensor,
        error_norms: Tensor,
    ) -> Tensor:
        """Compute candidate scores for all possible block positions.

        Score for position (r, c) = error_norm[r] * activation_norm[c]
        High error at output r AND high activation at input c suggests
        connecting r to c would help learning.

        Args:
            activation_norms: Input activation L2 norms [C]
            error_norms: Output gradient L2 norms [R]

        Returns:
            Candidate score matrix [R, C]
        """
        # Outer product: [R] x [C] -> [R, C]
        # error_norms[:, None] is [R, 1], activation_norms[None, :] is [1, C]
        return torch.outer(error_norms, activation_norms)

    def select_top_k(
        self,
        current_scores: Tensor,
        candidate_scores: Tensor,
        col_indices: Tensor,
        generator: Optional[torch.Generator] = None,
        block_ages: Optional[Tensor] = None,
        row_swap_cap_fraction: float = 0.2,
        layer_swap_budget_fraction: float = 0.05,
        reserved_mask: Optional[Tensor] = None,
        reserve_weight: float = 0.5,
    ) -> Tuple[Tensor, List[Tuple[int, int]], List[int]]:
        """Select top-K blocks per row with epsilon-greedy exploration.

        For each row:
        1. Get scores for current blocks from current_scores
        2. Get scores for candidate positions from candidate_scores
        3. Normalize candidate scores to match current score scale (Fix 2)
        4. T037: Apply reserved column bonus AFTER normalization (FR-011)
        5. Apply age-based protection for mature blocks (Fix 5)
        6. With probability epsilon, add random noise for exploration
        7. Apply swap_threshold: only swap if candidate is 1.5x better
        8. Select top-K scoring positions

        Args:
            current_scores: Gradient EMA for current blocks [R, K]
            candidate_scores: Outer product scores [R, C]
            col_indices: Current column indices [R, K]
            generator: RNG for deterministic exploration
            block_ages: Optional ages of current blocks [R, K] for age protection
            reserved_mask: Optional boolean mask [C] indicating reserved (underutilized) columns
            reserve_weight: Bonus multiplier for reserved columns (default 0.5 = 50% bonus)

        Returns:
            Tuple of:
            - new_col_indices: Updated column indices [R, K]
            - pruned_positions: List of (row, slot) that were pruned
            - grown_columns: List of new column indices added
        """
        R, K = current_scores.shape
        C = candidate_scores.shape[1]
        device = current_scores.device
        dtype = current_scores.dtype

        # Ensure col_indices is int64 (scatter_ requires it in newer PyTorch)
        col_indices = col_indices.to(dtype=torch.int64)

        # Fix 2: Normalize candidate scores to match current score scale
        # This prevents score magnitude mismatch from causing unstable swaps
        current_positive = current_scores[current_scores > 0]
        if current_positive.numel() > 0:
            current_mean = current_positive.mean()
        else:
            current_mean = torch.tensor(1.0, device=device, dtype=dtype)

        candidate_mean = candidate_scores.mean()
        if candidate_mean > 0 and current_mean > 0:
            scale_factor = current_mean / candidate_mean
            candidate_scores = candidate_scores * scale_factor

        # T037 (FR-011): Apply reserved column bonus AFTER normalization
        # Reserved columns (underutilized) get bonus scoring to encourage new tasks
        # to use them rather than fighting for established pathways.
        # IMPORTANT: This bonus is applied AFTER normalization, not before, so it
        # doesn't get washed out by the scale_factor adjustment above.
        if reserved_mask is not None and reserve_weight > 0:
            # Broadcast reserved_mask [C] to [R, C] for element-wise multiplication
            # reserved_bonus_mask is 1.0 for non-reserved, (1 + reserve_weight) for reserved
            reserved_bonus = torch.ones(C, device=device, dtype=dtype)
            reserved_bonus[reserved_mask] = 1.0 + reserve_weight
            # Apply bonus to candidate scores (broadcasts across rows)
            candidate_scores = candidate_scores * reserved_bonus.unsqueeze(0)

        # Fix 5: Age protection - boost scores of older blocks to prevent churn
        # Blocks older than 5 topology steps get increasing protection (up to 2x boost)
        if block_ages is not None:
            age_bonus = torch.clamp(block_ages.float() / 5.0, min=0.0, max=2.0)
            # Apply age bonus: older blocks get up to 2x score boost
            current_scores = current_scores * (1.0 + age_bonus * 0.5)

        # Create mask for active columns per row [R, C]
        # old_mask[r, c] = True if column c is currently active in row r
        old_mask = torch.zeros((R, C), dtype=torch.bool, device=device)
        old_mask.scatter_(1, col_indices, True)

        # Apply swap threshold: boost current block scores by swap_threshold factor
        # This makes it harder for candidates to beat them (candidate must be 1.5x better)
        boosted_current = current_scores * self.swap_threshold

        # Build combined score matrix [R, C]:
        # - Active positions: use boosted current_scores
        # - Inactive positions: use candidate_scores
        combined_scores = torch.full((R, C), float("-inf"), device=device, dtype=dtype)
        combined_scores.scatter_(1, col_indices, boosted_current)
        combined_scores = torch.where(old_mask, combined_scores, candidate_scores)

        # Epsilon-greedy exploration: with probability epsilon, add random noise
        if self.exploration_epsilon > 0:
            # Generate random mask for which positions get exploration noise
            explore_mask = (
                torch.rand((R, C), device=device, generator=generator) < self.exploration_epsilon
            )

            # Add large random noise to selected positions to potentially force selection
            # The noise should be large enough to override normal scoring
            max_score = (
                combined_scores[combined_scores != float("-inf")].max().item()
                if (combined_scores != float("-inf")).any()
                else 1.0
            )
            noise = torch.rand((R, C), device=device, generator=generator) * max_score * 2

            # Only apply noise to inactive positions (exploring new blocks, not keeping bad old ones)
            explore_mask = explore_mask & ~old_mask
            combined_scores = torch.where(explore_mask, noise, combined_scores)

        # Select top-K per row
        _, new_col_indices = torch.topk(combined_scores, K, dim=1, sorted=True)
        new_col_indices = new_col_indices.sort(dim=1).values  # Keep sorted for consistency

        # ============================================================
        # T022/T023: Apply per-row swap cap and per-layer budget
        # ============================================================
        # Calculate limits (ensure minimum of 1 for both to avoid zero-budget deadlock)
        max_swaps_per_row = max(1, int(row_swap_cap_fraction * K))
        layer_swap_budget = max(1, int(layer_swap_budget_fraction * R * K))

        # Create initial mask for proposed new topology
        proposed_new_mask = torch.zeros((R, C), dtype=torch.bool, device=device)
        proposed_new_mask.scatter_(1, new_col_indices, True)

        # Identify swaps per row: columns being added (swap_in)
        swap_in_mask = proposed_new_mask & ~old_mask  # [R, C]
        swaps_per_row = swap_in_mask.sum(dim=1)  # [R]

        # Check if any row exceeds caps or if layer budget could be exceeded
        total_proposed_swaps = swaps_per_row.sum().item()
        any_row_exceeds_cap = (swaps_per_row > max_swaps_per_row).any().item()

        if any_row_exceeds_cap or total_proposed_swaps > layer_swap_budget:
            # Need to limit swaps - process rows in random order
            row_order = torch.randperm(R, device=device, generator=generator)
            modified_indices = new_col_indices.clone()
            total_swaps = 0
            budget_exhausted = False

            for r in row_order.tolist():
                if budget_exhausted:
                    # Budget exhausted - revert this row to old indices
                    modified_indices[r] = col_indices[r]
                    continue

                # Get column sets for this row
                old_cols_tensor = col_indices[r]
                new_cols_tensor = modified_indices[r]
                old_cols = set(old_cols_tensor.tolist())
                new_cols = set(new_cols_tensor.tolist())

                retained = old_cols & new_cols
                swap_in = new_cols - old_cols  # columns being added
                swap_out = old_cols - new_cols  # columns being removed
                num_swaps = len(swap_in)

                if num_swaps == 0:
                    continue  # No swaps in this row, nothing to limit

                # Check layer budget (SOFT - exit early when exhausted)
                remaining_budget = layer_swap_budget - total_swaps
                if remaining_budget <= 0:
                    # Budget exhausted - revert this row
                    modified_indices[r] = col_indices[r]
                    budget_exhausted = True
                    continue

                # Calculate allowed swaps for this row:
                # - Per-row cap is HARD (never exceed)
                # - Layer budget is SOFT (best effort)
                row_allowed = min(max_swaps_per_row, remaining_budget, num_swaps)

                if row_allowed < num_swaps:
                    # Need to reduce swaps - keep only the best ones by candidate score
                    swap_in_list = list(swap_in)
                    swap_out_list = list(swap_out)

                    # Sort swap_in columns by their candidate score (descending)
                    swap_in_scores = [candidate_scores[r, c].item() for c in swap_in_list]
                    sorted_pairs = sorted(zip(swap_in_scores, swap_in_list), reverse=True)
                    keep_cols = [c for _, c in sorted_pairs[:row_allowed]]

                    # Restore old columns to fill the gap (the ones we're NOT swapping out)
                    restore_count = num_swaps - row_allowed
                    restore_cols = swap_out_list[:restore_count]

                    # Build final column set for this row
                    final_cols = list(retained) + keep_cols + restore_cols
                    final_cols.sort()  # Keep sorted for consistency

                    # Safety check: ensure exactly K elements
                    # Edge case: duplicate column indices in input can cause count mismatches
                    if len(final_cols) != K:
                        # Fill missing from swap_out if needed
                        if len(final_cols) < K:
                            additional = [
                                c
                                for c in swap_out_list[restore_count:]
                                if c not in final_cols
                            ][: K - len(final_cols)]
                            final_cols.extend(additional)
                            final_cols.sort()
                        # If still wrong count, revert to original (safe fallback)
                        if len(final_cols) != K:
                            modified_indices[r] = col_indices[r]
                            continue

                    modified_indices[r] = torch.tensor(
                        final_cols, device=device, dtype=col_indices.dtype
                    )
                    num_swaps = row_allowed

                total_swaps += num_swaps

                # Check if budget now exhausted for early exit
                if total_swaps >= layer_swap_budget:
                    budget_exhausted = True

            new_col_indices = modified_indices

        # Vectorized pruned/grown computation on GPU
        # Create mask for new topology (after swap caps applied)
        new_mask = torch.zeros((R, C), dtype=torch.bool, device=device)
        new_mask.scatter_(1, new_col_indices, True)

        # Pruned: was in old but not in new
        # We need (row, slot) pairs where slot is the position in col_indices
        # pruned_col_mask[r, c] = True if column c was pruned from row r
        pruned_col_mask = old_mask & ~new_mask

        # For each row, find which slots were pruned by checking if col_indices[r, slot] was pruned
        # Gather the pruned status for each slot position
        pruned_slot_mask = torch.gather(pruned_col_mask, 1, col_indices)  # [R, K]

        # Grown: in new but not in old
        grown_col_mask = new_mask & ~old_mask

        # Single GPU->CPU sync: transfer only the masks we need
        # Only sync if there are any changes to report
        if pruned_slot_mask.any() or grown_col_mask.any():
            # Get pruned positions as (row, slot) tuples
            pruned_indices = pruned_slot_mask.nonzero(as_tuple=False)  # [num_pruned, 2]
            pruned_positions = [
                (int(row), int(slot)) for row, slot in pruned_indices.tolist()
            ]

            # Get grown column indices
            grown_indices = grown_col_mask.nonzero(as_tuple=False)  # [num_grown, 2]
            grown_columns = [int(col) for _, col in grown_indices.tolist()]
        else:
            pruned_positions = []
            grown_columns = []

        return new_col_indices, pruned_positions, grown_columns

    def compute_column_entropy(self, col_indices: Tensor) -> float:
        """Compute normalized entropy of column usage across all rows.

        High entropy means columns are used uniformly (good diversity).
        Low entropy means some columns are overused (poor diversity).

        Args:
            col_indices: Column indices [R, K]

        Returns:
            Normalized entropy in [0, 1]
        """
        # Count usage frequency of each column
        # col_indices is [R, K], we want to count how often each column index appears
        flat_indices = col_indices.flatten()  # [R * K]

        # Use bincount to count occurrences, with minlength=C
        counts = torch.bincount(flat_indices.to(torch.int64), minlength=self.C)  # [C]

        # Convert to probabilities (exclude zero counts from entropy calculation)
        total = counts.sum().float()
        if total == 0:
            return 0.0

        # Filter out zero counts to avoid log(0)
        nonzero_mask = counts > 0
        nonzero_counts = counts[nonzero_mask].float()

        # Compute probabilities
        probs = nonzero_counts / total

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs))

        # Normalize by max entropy (log(C)) to get value in [0, 1]
        if self.C <= 1:
            return 1.0  # Edge case: only one column means max entropy

        max_entropy = torch.log(torch.tensor(float(self.C), device=col_indices.device))
        normalized_entropy = (entropy / max_entropy).item()

        return normalized_entropy

    def should_swap(
        self,
        current_score: float,
        candidate_score: float,
        block_age: int,
    ) -> bool:
        """Determine if a swap should occur based on scores and age.

        A swap occurs if:
        - candidate_score > current_score * swap_threshold

        Note: block_age parameter is reserved for future use (age-based protection)
        but is currently ignored per task specification.

        Args:
            current_score: Score of current block
            candidate_score: Score of candidate position
            block_age: Age of current block in topology steps (reserved for future)

        Returns:
            True if swap should occur
        """
        # Simple threshold check: candidate must be swap_threshold times better
        return candidate_score > current_score * self.swap_threshold


def compute_gradient_frobenius_norms(grad: Tensor) -> Tensor:
    """Compute Frobenius norm of gradients per block.

    Frobenius norm = sqrt(sum of squared elements) per block.

    Args:
        grad: Gradient tensor [R, K, B, B]

    Returns:
        Frobenius norms [R, K]
    """
    # Frobenius norm: sqrt(sum of squared elements) per block
    # grad shape is [R, K, B, B], we want output [R, K]
    # Sum over the last two dimensions (B, B), then sqrt
    return torch.sqrt(torch.sum(grad * grad, dim=(-2, -1)))


def initialize_scores(
    R: int,
    C: int,
    K: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initialize scoring tensors for a new layer.

    Args:
        R: Number of output block-rows
        C: Number of input block-columns
        K: Active blocks per row
        device: Target device for tensors
        dtype: Data type for float tensors (block_age is always int32)

    Returns:
        Tuple of:
        - block_score_ema: [R, K] initialized to zeros
        - activation_norm_acc: [C] initialized to zeros
        - error_norm_acc: [R] initialized to zeros
        - block_age: [R, K] initialized to zeros (int32)
    """
    # Use defaults if not specified
    if dtype is None:
        dtype = torch.float32

    block_score_ema = torch.zeros(R, K, device=device, dtype=dtype)
    activation_norm_acc = torch.zeros(C, device=device, dtype=dtype)
    error_norm_acc = torch.zeros(R, device=device, dtype=dtype)
    block_age = torch.zeros(R, K, device=device, dtype=torch.int32)

    return block_score_ema, activation_norm_acc, error_norm_acc, block_age
