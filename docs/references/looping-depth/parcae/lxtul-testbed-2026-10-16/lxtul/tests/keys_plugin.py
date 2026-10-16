"""Run an existing LXTUL test file with speed keys switched on, without editing it.

    LXTUL_KEYS='{"compact_slots": true, "ce_rows": true}' \
        python -m pytest -q -p lxtul.tests.keys_plugin lxtul/tests/test_geometry.py

Every `LXTULParcae(config, tul)` built during the session gets `dataclasses.replace(tul, **keys)`.
A session-scoped fixture, so it is in place before the module-scoped models those tests build.
"""
from __future__ import annotations

import dataclasses
import json
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _lxtul_keys():
    keys = json.loads(os.environ.get("LXTUL_KEYS") or "{}")
    if not keys:
        yield
        return
    import lxtul.model as lm
    real = lm.LXTULParcae.__init__

    def init(self, config, tul):
        real(self, config, dataclasses.replace(tul, **keys))
    mp = pytest.MonkeyPatch()
    mp.setattr(lm.LXTULParcae, "__init__", init)
    print(f"\n[keys_plugin] LXTUL keys on: {keys}")
    yield
    mp.undo()
