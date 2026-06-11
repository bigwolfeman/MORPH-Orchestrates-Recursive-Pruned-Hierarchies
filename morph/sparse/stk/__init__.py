# Vendored stk (Apache-2.0, stanford-futuredata/stk) — see NOTICE for provenance
# and the MORTAR modification list. Public surface mirrors upstream `import stk`.
from morph.sparse.stk.matrix import Matrix
from morph.sparse.stk.ops.linear_ops import dds, dsd, sdd
from morph.sparse.stk.ops.matrix_ops import row_indices, to_dense, to_sparse, ones_like

__all__ = ["Matrix", "dds", "dsd", "sdd", "row_indices", "to_dense", "to_sparse", "ones_like"]
