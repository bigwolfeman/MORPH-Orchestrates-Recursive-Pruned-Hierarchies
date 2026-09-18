"""Contract test for `core_depth_sweep.py`'s frozen-reader warning.

Measured 2026-09-18 (480 rows, three code-target arms): a code-target arm whose coda is
FROZEN reports a `ce_tokens` K-curve that tracks how GENERIC the predicted cell becomes
with depth, not how well it matches the code. The frozen coda was trained on noised TRUE
codes and prefers no cell at all (`zero - own` = -4.64 nats), so a cell drifting toward
the corpus mean reads as an improvement and a cell becoming more span-specific reads as a
regression:

    arm A  K1-K6 +0.0323   pred.zbar 0.3081 -> 0.3225   (blander with depth, CE "improves")
    prog   K1-K6 -0.1173   pred.zbar 0.3554 -> 0.3051   (sharper with depth, CE degrades)
    uf     K1-K6 +0.0047   coda TRAINED; CE minimum and cosine maximum both at depth 6

So the sweep must say so, and it must say so on exactly the arms where it is true. The
discriminator is whether `coda.` appears in `training.train_only`. These tests fail if the
predicate is inverted, widened to every TUL arm, or narrowed away.

Writeup: lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md
"""
import contextlib
import io
import sys
from pathlib import Path

import pytest

_DIV = Path(__file__).resolve().parents[1] / "lab" / "divergence"
if str(_DIV) not in sys.path:  # the probe uses bare imports (`from _build import ...`)
    sys.path.insert(0, str(_DIV))

from _build import build_cfg  # noqa: E402
from core_depth_sweep import warn_if_frozen_reader  # noqa: E402


def _warn_text(cfg_name):
    cfg = build_cfg(cfg_name, ["model.use_kernels=false"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        warn_if_frozen_reader(cfg, cfg_name)
    return buf.getvalue()


@pytest.mark.parametrize(
    "cfg_name, expect_warning",
    [
        ("tul_code_target", True),        # the coda is frozen: the K-curve is genericity
        ("tul_code_target_prog", True),   # same, plus progressive_p
        ("tul_code_target_uf", False),    # `coda.` IS in train_only: the reading is valid
        ("tul_a2", False),                # not a code-target arm at all
    ],
)
def test_warns_exactly_when_the_reader_is_frozen(cfg_name, expect_warning):
    warned = "[WARNING]" in _warn_text(cfg_name)
    assert warned is expect_warning, (
        f"{cfg_name}: warned={warned}, expected={expect_warning}. The predicate is "
        f"`tul.code_target` set AND no `coda.` prefix in training.train_only."
    )


def test_the_warning_names_the_companion_instrument():
    """A warning that does not say what to do instead is theatre: the reader needs to be
    sent to the probe that distinguishes genericity from code match."""
    text = _warn_text("tul_code_target")
    assert "pred.zbar" in text, "the warning must name the quantity that decides it"
    assert "code_target_mean_probe" in text, "the warning must name the companion probe"


def test_unfreezing_the_coda_is_what_silences_it():
    """The discriminator is the coda entry specifically, not merely a longer train_only."""
    frozen = build_cfg("tul_code_target", ["model.use_kernels=false"])
    adapted = build_cfg("tul_code_target_uf", ["model.use_kernels=false"])
    frozen_list = [str(p) for p in frozen.training.train_only]
    adapted_list = [str(p) for p in adapted.training.train_only]
    assert not any(p.startswith("coda.") for p in frozen_list)
    assert any(p.startswith("coda.") for p in adapted_list)
