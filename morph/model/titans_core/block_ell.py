"""Block-ELL sparse tensor format for block-sparse operations.

Block-ELL (Block-ELLPACK) stores sparse matrices as:
- values: Dense [R, K, B, B] tensor of block values
- col_indices: [R, K] tensor of column indices per block

This format is optimized for:
- Fixed number of blocks per row (K) for regular memory access
- Tensor core operations on B x B tiles
- Efficient CUDA kernel execution with coalesced memory access

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass
class BlockELLConfig:
    """Configuration for Block-ELL format.

    Args:
        R: Number of output block-rows (out_features // tile_size)
        C: Number of input block-columns (in_features // tile_size)
        K: Number of active blocks per row
        B: Block/tile size (default 16 for WMMA compatibility)

    Properties:
        out_features: R * B
        in_features: C * B
        total_blocks: R * K
        total_parameters: R * K * B * B
        density: K / C
    """

    R: int
    C: int
    K: int
    B: int = 16

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.K > self.C:
            raise ValueError(f"K ({self.K}) cannot exceed C ({self.C})")
        if self.K < 1:
            raise ValueError(f"K ({self.K}) must be at least 1")
        if self.R < 1:
            raise ValueError(f"R ({self.R}) must be at least 1")
        if self.C < 1:
            raise ValueError(f"C ({self.C}) must be at least 1")
        if self.B < 1:
            raise ValueError(f"B ({self.B}) must be at least 1")

    @property
    def out_features(self) -> int:
        """Output dimension: R * B."""
        return self.R * self.B

    @property
    def in_features(self) -> int:
        """Input dimension: C * B."""
        return self.C * self.B

    @property
    def total_blocks(self) -> int:
        """Total number of active blocks: R * K."""
        return self.R * self.K

    @property
    def total_parameters(self) -> int:
        """Total number of weight parameters: R * K * B * B."""
        return self.R * self.K * self.B * self.B

    @property
    def density(self) -> float:
        """Fraction of active blocks: K / C."""
        return self.K / self.C


@dataclass
class BlockELLTensor:
    """Block-ELL sparse tensor representation.

    This dataclass holds the core data for a block-sparse matrix in
    Block-ELL format. It does not own the tensors (no copying).

    Attributes:
        values: Weight values tensor [R, K, B, B]
        col_indices: Column index for each block [R, K] (int32)
        config: BlockELLConfig with dimensions

    Example:
        >>> config = BlockELLConfig(R=160, C=40, K=20, B=16)
        >>> values = torch.randn(160, 20, 16, 16)
        >>> col_indices = torch.randint(0, 40, (160, 20), dtype=torch.int32)
        >>> block_ell = BlockELLTensor(values=values, col_indices=col_indices, config=config)
    """

    values: Tensor  # [R, K, B, B]
    col_indices: Tensor  # [R, K] int32
    config: BlockELLConfig

    def __post_init__(self) -> None:
        """Validate tensor shapes match config."""
        cfg = self.config
        expected_values_shape = (cfg.R, cfg.K, cfg.B, cfg.B)
        expected_indices_shape = (cfg.R, cfg.K)

        if self.values.shape != expected_values_shape:
            raise ValueError(
                f"values shape {self.values.shape} doesn't match "
                f"expected {expected_values_shape}"
            )
        if self.col_indices.shape != expected_indices_shape:
            raise ValueError(
                f"col_indices shape {self.col_indices.shape} doesn't match "
                f"expected {expected_indices_shape}"
            )


def create_block_ell_from_dense(
    dense: Tensor,
    tile_size: int = 16,
    density: float = 0.5,
    selection_method: str = "magnitude",
) -> BlockELLTensor:
    """Convert dense weight matrix to Block-ELL format.

    Selects top-K blocks per row based on the selection method.

    Args:
        dense: Dense weight matrix [out_features, in_features]
        tile_size: Block size B (must divide both dimensions)
        density: Fraction of blocks to keep per row
        selection_method: How to select blocks ("magnitude" or "random")

    Returns:
        BlockELLTensor with the sparse representation

    Raises:
        ValueError: If dimensions not divisible by tile_size
    """
    # Note: selection_method is currently ignored (only magnitude-based is implemented)
    values, col_indices, R, C, K, B = from_dense(dense, tile_size, density)
    config = BlockELLConfig(R=R, C=C, K=K, B=B)
    return BlockELLTensor(values=values, col_indices=col_indices.to(torch.int32), config=config)


def block_ell_to_dense(block_ell: BlockELLTensor) -> Tensor:
    """Convert Block-ELL tensor back to dense format.

    Args:
        block_ell: Block-ELL sparse tensor

    Returns:
        Dense weight matrix [out_features, in_features]
    """
    cfg = block_ell.config
    return to_dense(
        values=block_ell.values,
        col_indices=block_ell.col_indices,
        R=cfg.R,
        C=cfg.C,
        K=cfg.K,
        B=cfg.B,
    )


def validate_block_ell_topology(block_ell: BlockELLTensor) -> Tuple[bool, Optional[str]]:
    """Validate Block-ELL topology constraints.

    Checks:
    - All col_indices are in valid range [0, C)
    - Each row has unique column indices (no duplicates)

    Args:
        block_ell: Block-ELL tensor to validate

    Returns:
        Tuple of (is_valid, error_message or None)
    """
    cfg = block_ell.config
    # Create a BlockELLFormat to use its validate() method
    block_ell_format = BlockELLFormat(
        R=cfg.R,
        C=cfg.C,
        K=cfg.K,
        B=cfg.B,
        values=block_ell.values,
        col_indices=block_ell.col_indices,
    )
    return block_ell_format.validate()


def initialize_block_ell_topology(
    config: BlockELLConfig,
    method: str = "random",
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Initialize column indices for Block-ELL topology.

    Args:
        config: BlockELLConfig specifying dimensions
        method: Initialization method ("random", "strided", "first_k")
        device: Target device for the tensor
        generator: RNG for deterministic initialization

    Returns:
        col_indices tensor [R, K] with int32 dtype

    Raises:
        ValueError: If method is not supported
    """
    if method == "random":
        col_indices = create_random_topology(
            R=config.R,
            C=config.C,
            K=config.K,
            generator=generator,
            device=device,
            dtype=torch.int32,
        )
        return col_indices
    elif method == "strided":
        # Strided: evenly spaced columns
        col_indices = torch.zeros(config.R, config.K, dtype=torch.int32, device=device)
        stride = config.C // config.K
        for k in range(config.K):
            col_indices[:, k] = min(k * stride, config.C - 1)
        return col_indices
    elif method == "first_k":
        # First K: select columns 0, 1, ..., K-1
        col_indices = torch.arange(config.K, dtype=torch.int32, device=device)
        col_indices = col_indices.unsqueeze(0).expand(config.R, -1).clone()
        return col_indices
    else:
        raise ValueError(
            f"Unknown initialization method: {method}. Supported: random, strided, first_k"
        )


@dataclass
class BlockELLFormat:
    """Block-ELL sparse tensor format for CMS dynamic block-sparse operations.

    This dataclass encapsulates both the configuration and data for a block-sparse
    matrix in Block-ELL format. It provides validation to ensure data integrity.

    Attributes:
        R: Number of output block-rows (out_features // tile_size)
        C: Number of input block-columns (in_features // tile_size)
        K: Number of active blocks per row
        B: Block/tile size (default 16 for WMMA tensor core compatibility)
        values: Weight values tensor [R, K, B, B]
        col_indices: Column index for each block [R, K] (int64)

    Validation Rules:
        - K <= C (can't have more active blocks than columns)
        - All col_indices[r] values must be unique within each row r
        - All col_indices values must be in range [0, C)
        - values.shape must match [R, K, B, B]
        - col_indices.shape must match [R, K]

    Example:
        >>> format = BlockELLFormat(
        ...     R=10, C=20, K=5, B=16,
        ...     values=torch.randn(10, 5, 16, 16),
        ...     col_indices=torch.randint(0, 20, (10, 5))
        ... )
        >>> is_valid, error = format.validate()
    """

    R: int
    C: int
    K: int
    B: int
    values: Tensor  # [R, K, B, B]
    col_indices: Tensor  # [R, K]

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate the Block-ELL format constraints.

        Checks:
        1. R, C, K, B are all positive
        2. K <= C (can't have more active blocks than columns)
        3. values.shape matches [R, K, B, B]
        4. col_indices.shape matches [R, K]
        5. All col_indices are in valid range [0, C)
        6. Each row has unique column indices (no duplicates within a row)

        Returns:
            Tuple of (is_valid, error_message or None)
        """
        # Check dimension constraints
        if self.R < 1:
            return False, f"R ({self.R}) must be at least 1"
        if self.C < 1:
            return False, f"C ({self.C}) must be at least 1"
        if self.K < 1:
            return False, f"K ({self.K}) must be at least 1"
        if self.B < 1:
            return False, f"B ({self.B}) must be at least 1"
        if self.K > self.C:
            return False, f"K ({self.K}) cannot exceed C ({self.C})"

        # Check values tensor shape
        expected_values_shape = (self.R, self.K, self.B, self.B)
        if self.values.shape != expected_values_shape:
            return False, (
                f"values shape {tuple(self.values.shape)} doesn't match "
                f"expected {expected_values_shape}"
            )

        # Check col_indices tensor shape
        expected_indices_shape = (self.R, self.K)
        if self.col_indices.shape != expected_indices_shape:
            return False, (
                f"col_indices shape {tuple(self.col_indices.shape)} doesn't match "
                f"expected {expected_indices_shape}"
            )

        # Check col_indices are in valid range [0, C)
        if self.col_indices.numel() > 0:
            min_idx = self.col_indices.min().item()
            max_idx = self.col_indices.max().item()
            if min_idx < 0:
                return False, f"col_indices contains negative value: {min_idx}"
            if max_idx >= self.C:
                return False, f"col_indices contains out-of-range value: {max_idx} >= C ({self.C})"

        # Check uniqueness per row: each row's col_indices must be unique
        for r in range(self.R):
            row_indices = self.col_indices[r]
            unique_count = row_indices.unique().numel()
            if unique_count != self.K:
                return False, (
                    f"Row {r} has duplicate column indices: "
                    f"{unique_count} unique values instead of {self.K}"
                )

        return True, None

    @property
    def out_features(self) -> int:
        """Output dimension: R * B."""
        return self.R * self.B

    @property
    def in_features(self) -> int:
        """Input dimension: C * B."""
        return self.C * self.B

    @property
    def total_blocks(self) -> int:
        """Total number of active blocks: R * K."""
        return self.R * self.K

    @property
    def total_parameters(self) -> int:
        """Total number of weight parameters: R * K * B * B."""
        return self.R * self.K * self.B * self.B

    @property
    def density(self) -> float:
        """Fraction of active blocks: K / C."""
        return self.K / self.C


def create_random_topology(
    R: int,
    C: int,
    K: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """Create random column indices for Block-ELL topology.

    Generates column indices where each row has K unique values randomly
    selected from the range [0, C). This ensures no duplicate columns
    within any single row.

    Args:
        R: Number of output block-rows
        C: Number of input block-columns
        K: Number of active blocks per row (must be <= C)
        generator: Optional RNG for reproducible initialization
        device: Target device for the tensor
        dtype: Data type for indices (default torch.long)

    Returns:
        col_indices tensor [R, K] with unique indices per row

    Raises:
        ValueError: If K > C (can't select K unique values from C)

    Example:
        >>> indices = create_random_topology(R=4, C=8, K=3)
        >>> indices.shape
        torch.Size([4, 3])
        >>> # Each row has 3 unique values in [0, 8)
    """
    if K > C:
        raise ValueError(f"K ({K}) cannot exceed C ({C})")
    if K < 1:
        raise ValueError(f"K ({K}) must be at least 1")
    if R < 1:
        raise ValueError(f"R ({R}) must be at least 1")
    if C < 1:
        raise ValueError(f"C ({C}) must be at least 1")

    # Generate random permutations and take first K indices for each row
    # This guarantees uniqueness within each row
    col_indices = torch.zeros(R, K, dtype=dtype, device=device)

    for r in range(R):
        # Generate a random permutation of [0, C) and take first K
        perm = torch.randperm(C, generator=generator, device=device)
        col_indices[r] = perm[:K]

    return col_indices


def to_dense(
    values: Tensor,
    col_indices: Tensor,
    R: int,
    C: int,
    K: int,
    B: int,
) -> Tensor:
    """Convert Block-ELL format to dense matrix.

    Takes block values and column indices and reconstructs the full
    dense weight matrix by placing each block at its corresponding
    position.

    Args:
        values: Block values tensor [R, K, B, B]
        col_indices: Column indices for each block [R, K]
        R: Number of output block-rows
        C: Number of input block-columns
        K: Number of active blocks per row
        B: Block/tile size

    Returns:
        Dense weight matrix [R*B, C*B] (out_features x in_features)

    Example:
        >>> values = torch.randn(4, 3, 16, 16)  # 4 rows, 3 blocks each
        >>> col_indices = torch.tensor([[0, 2, 5], [1, 3, 4], [0, 1, 2], [2, 3, 4]])
        >>> dense = to_dense(values, col_indices, R=4, C=8, K=3, B=16)
        >>> dense.shape
        torch.Size([64, 128])
    """
    # Create output dense matrix
    out_features = R * B
    in_features = C * B
    dense = torch.zeros(out_features, in_features, dtype=values.dtype, device=values.device)

    # Place each block at its position
    for r in range(R):
        for k in range(K):
            c = col_indices[r, k].item()
            row_start = r * B
            row_end = row_start + B
            col_start = c * B
            col_end = col_start + B
            dense[row_start:row_end, col_start:col_end] = values[r, k]

    return dense


def from_dense(
    dense: Tensor,
    tile_size: int = 16,
    density: float = 0.5,
) -> Tuple[Tensor, Tensor, int, int, int, int]:
    """Convert dense matrix to Block-ELL format using magnitude-based pruning.

    Divides the dense matrix into blocks and selects the top-K blocks per row
    based on Frobenius norm (magnitude). This preserves the most important
    weights while achieving the target sparsity.

    Args:
        dense: Dense weight matrix [out_features, in_features]
        tile_size: Block size B (must divide both dimensions evenly)
        density: Fraction of blocks to keep per row (0 < density <= 1)

    Returns:
        Tuple of (values, col_indices, R, C, K, B):
            - values: Block values tensor [R, K, B, B]
            - col_indices: Column indices [R, K]
            - R: Number of output block-rows
            - C: Number of input block-columns
            - K: Number of active blocks per row
            - B: Block/tile size

    Raises:
        ValueError: If dimensions not divisible by tile_size

    Example:
        >>> dense = torch.randn(64, 128)
        >>> values, col_indices, R, C, K, B = from_dense(dense, tile_size=16, density=0.5)
        >>> # R=4, C=8, K=4, B=16 (50% of 8 columns = 4 blocks per row)
    """
    out_features, in_features = dense.shape
    B = tile_size

    # Validate dimensions are divisible by tile_size
    if out_features % B != 0:
        raise ValueError(f"out_features ({out_features}) must be divisible by tile_size ({B})")
    if in_features % B != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by tile_size ({B})")

    R = out_features // B
    C = in_features // B
    K = max(1, int(C * density))  # At least 1 block per row

    # Compute block magnitudes (Frobenius norm of each block)
    # Reshape to [R, C, B, B] and compute norm over last two dims
    blocks = dense.view(R, B, C, B).permute(0, 2, 1, 3)  # [R, C, B, B]
    block_norms = blocks.norm(dim=(2, 3))  # [R, C] - Frobenius norm of each block

    # For each row, select top-K blocks by magnitude
    values = torch.zeros(R, K, B, B, dtype=dense.dtype, device=dense.device)
    col_indices = torch.zeros(R, K, dtype=torch.long, device=dense.device)

    for r in range(R):
        # Get top-K column indices by magnitude for this row
        row_norms = block_norms[r]  # [C]
        _, top_k_indices = torch.topk(row_norms, K)
        top_k_indices_sorted = top_k_indices.sort().values  # Keep sorted for consistency

        col_indices[r] = top_k_indices_sorted
        for k_idx, c in enumerate(top_k_indices_sorted):
            values[r, k_idx] = blocks[r, c]

    return values, col_indices, R, C, K, B
