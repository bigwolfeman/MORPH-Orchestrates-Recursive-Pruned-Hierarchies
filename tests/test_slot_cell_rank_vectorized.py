"""The within-slot rank reading (`slot_cell_eff_rank`, `slot_cell_pairwise_cos`) is
vectorised on the device. It must equal the per-slot loop it replaced (one
`effective_rank` + one normalised Gram per valid slot), which stalled the GPU for
~3.7 min per val on a register arm (2026-09-13). Reference loop reimplemented here."""
from __future__ import annotations

import torch

from morph.model.fm_planner import effective_rank
from tests.test_tul_slot_register import _batch, _model  # the register fixtures


def _reference(z: torch.Tensor, slot_valid: torch.Tensor, m: int):
    B, SM, C = z.shape
    S = SM // m
    zc = z.reshape(B, S, m, C).cpu()
    ranks, coss = [], []
    for b in range(B):
        for s in range(S):
            if not bool(slot_valid[b, s]):
                continue
            cells = zc[b, s]
            ranks.append(effective_rank(cells.unsqueeze(0), torch.ones(1, m, dtype=torch.bool)))
            n = torch.nn.functional.normalize(cells, dim=-1)
            g = n @ n.T
            iu = torch.triu_indices(m, m, offset=1)
            coss.append(float(g[iu[0], iu[1]].mean()))
    return sum(ranks) / len(ranks), sum(coss) / len(coss)


def test_vectorised_within_slot_rank_equals_the_per_slot_loop():
    m = _model(M=4, seed=11).eval()
    _ids0, inp, _lab, layout = _batch(M=4, seed=5)
    with torch.no_grad():
        out = m.tul_slot_state_probe(inp, layout)
        # the same states the probe read, rebuilt the way the probe builds them
        fkw, freset, _c, _cr = m._tul_tg_kwargs(layout)
        x, x0, bg = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        h = m._tul_core(x, x0, bg, layout, input_ids=inp)[1]
        z = m._readout(h).float()
    er, cos = _reference(z, layout.slot_valid, 4)
    assert abs(out["slot_cell_eff_rank"] - er) < 1e-4, (out["slot_cell_eff_rank"], er)
    assert abs(out["slot_cell_pairwise_cos"] - cos) < 1e-5, (out["slot_cell_pairwise_cos"], cos)
    assert out["slot_cells"] == 4.0
    assert 1.0 < out["slot_cell_eff_rank"] <= 4.0 and abs(cos) < 1.0   # non-vacuous
