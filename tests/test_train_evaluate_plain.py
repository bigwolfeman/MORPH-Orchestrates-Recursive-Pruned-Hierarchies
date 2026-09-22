"""`evaluate()` on a PLAIN model (no TUL): the regression test for the 2026-09-21 reach-split
smokes, which died at the first eval with `UnboundLocalError: _tul_cfg` because the code-rank
abort (d0a2308) read a local that only the TUL branch assigned. Every plain-model arm from
2026-09-15 to 2026-09-21 would have died the same way at its first eval."""
import numpy as np
import torch

from morph.training.train import evaluate
from test_span_mask_leak import _tiny, _ids, V  # noqa: F401  (tests/ is on PYTHONPATH)
from morph.model.transformer import MORPHTransformer


def test_evaluate_runs_on_a_plain_model_with_extra():
    torch.manual_seed(0)
    model = MORPHTransformer(_tiny())
    model.eval()
    ids = torch.from_numpy(_ids(B=2, S=32))
    x, y = ids[:, :-1], ids[:, 1:]
    loader = iter([(x, y), (x, y)])
    extra: dict = {}
    loss, ppl = evaluate(model, torch.device("cpu"), loader, n_batches=2, tul=False, extra=extra)
    assert np.isfinite(loss) and ppl > 1.0
    assert "val/loss" not in extra or np.isfinite(extra["val/loss"])
    assert extra.get("val/ppl") is None or np.isfinite(extra["val/ppl"])
