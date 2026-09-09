"""The ONE ternary rule: codes and a scalar scale for one group of a weight.

Three paths turn a smooth latent weight into ``scale * codes`` and every one of them
calls :func:`ternary_codes_and_scale`, so the rule cannot drift between them:

- the training STE (``ternary_qat.TernarySTE``, mode ``norm_match``; the ``symmetric``
  mode keeps its own vectorised grouped path, which this function reproduces exactly),
- the carved MORTAR path (``layers.block_sparse.CMSBlockLinear._mortar_effective_data``,
  where QAT continues on ``mortar_data`` after ``carve()``),
- the deploy packer (``inference.deploy_quant``: ``extract_ternary_from_parametrized``
  and ``pack_mortar_ternary``).

Scale rules
-----------
``symmetric``  the BitNet b1.58 absmean rule: ``scale = mean|W_g|`` over ALL entries of the
               group, codes ``sign(w) * (|w| / scale > threshold)``. The zeroed entries are
               averaged into the scale, so a Gaussian-shaped layer's ternary weight has
               about 0.67 of the latent's Frobenius norm at threshold 0.5.
``norm_match`` the same codes, with ``scale = ||W_g||_F / sqrt(nnz_g)`` so that
               ``||scale * codes||_F == ||W_g||_F``. Ternary Weight Networks (Li et al.
               2016, arXiv:1605.04711, Eq. 5) is the nearest prior: the optimal scale is
               the mean magnitude over the NONZERO set; this is the root-mean-square
               analogue, which matches the norm exactly. No parameters, one scalar per
               group, so it exports in the same packed format as ``symmetric``.

Both are exportable; the learnable ``ttq`` / ``dual`` modes of ``TernarySTE`` are
training-only and are refused by the carve and the packer.

Measured reason for ``norm_match`` (2026-09-09, per-pass-strength panel): under the absmean
rule MORPH's looped core is a weak per-pass map (MLP branch out/in 0.19–0.24, loop
contribution K1−K6 0.033); the norm-matched scale restores the branch to 0.86–1.10 and the
contribution to 0.185. Record: ``lab/experiments/successes/2026-09-09-arc-per-pass-strength.md``;
decision: ``.agents/notes/implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md``.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

# Scale rules that a carved layer and the deploy packer can carry (one scalar per group,
# no learnable state). The order is the persisted integer code in the MORTAR
# ``mortar_ternary`` buffer: index 0 = symmetric, 1 = norm_match. Append only.
EXPORTABLE_SCALE_MODES: tuple[str, ...] = ("symmetric", "norm_match")


def encode_identity(raw: Tensor) -> Tensor:
    """The no-op scale encoder (the carved path and the packer keep the fp32 scale)."""
    return raw


def scale_mode_code(scale_mode: str) -> int:
    """Integer code of an exportable scale mode (for the persisted MORTAR buffer)."""
    if scale_mode not in EXPORTABLE_SCALE_MODES:
        raise ValueError(
            f"scale_mode {scale_mode!r} is not exportable; choices={EXPORTABLE_SCALE_MODES}. "
            "ttq / dual carry learnable scales that neither the carve nor the packed format "
            "represents.")
    return EXPORTABLE_SCALE_MODES.index(scale_mode)


def scale_mode_from_code(code: int | float) -> str:
    """Inverse of :func:`scale_mode_code`."""
    i = int(round(float(code)))
    if not 0 <= i < len(EXPORTABLE_SCALE_MODES):
        raise ValueError(f"unknown persisted scale-mode code {code!r}")
    return EXPORTABLE_SCALE_MODES[i]


def ternary_codes_and_scale(
    w: Tensor,
    threshold: float,
    scale_mode: str,
    encode_scale: Callable[[Tensor], Tensor] = encode_identity,
    scale_cap: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Codes and scalar scale of ONE group of a latent weight under ``scale_mode``.

    Parameters
    ----------
    w : Tensor
        The smooth latent weight of the group (any shape; detached inside). The codes
        come back in ``w``'s dtype and shape, the scale as a 0-dim tensor of that dtype.
    threshold : float
        ``|w| / mean|W_g| > threshold`` selects the nonzero codes (0.5 = Bonsai default).
    scale_mode : str
        One of :data:`EXPORTABLE_SCALE_MODES`.
    encode_scale : callable
        The scale-storage encoder (fp16 / int8 / pow2 in ``ternary_qat``; identity on the
        carved and packed paths). Applied to the absmean BEFORE the codes are cut, exactly
        as the ``symmetric`` STE does, and to the final scale.
    scale_cap : Tensor | None
        Optional 0-dim upper bound on the absmean (the ``symmetric`` mode's
        ``scale_clip_mult`` buffer). Applied after encoding, before the codes.

    Returns
    -------
    codes : Tensor in {-1, 0, +1}, ``w.dtype``, ``w.shape``
    scale : 0-dim Tensor, ``w.dtype``
    """
    if scale_mode not in EXPORTABLE_SCALE_MODES:
        raise ValueError(
            f"scale_mode {scale_mode!r} has no closed-form rule; "
            f"choices={EXPORTABLE_SCALE_MODES}")
    wd = w.detach()
    g = encode_scale(wd.abs().mean().clamp(min=1e-8).reshape(1))[0]
    if scale_cap is not None:
        g = torch.minimum(g, scale_cap)
    w_n = wd / g
    codes = torch.sign(w_n) * (w_n.abs() > threshold).to(wd.dtype)
    if scale_mode == "symmetric":
        return codes, g
    nnz = codes.ne(0).sum().clamp(min=1).to(wd.dtype)
    scale = encode_scale((wd.norm() / nnz.sqrt()).clamp(min=1e-8).reshape(1))[0]
    return codes, scale
