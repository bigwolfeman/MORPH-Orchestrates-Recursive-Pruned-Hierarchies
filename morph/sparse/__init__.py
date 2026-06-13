"""MORTAR — Macro-Orchestrated Routing and Tile-Aligned Recompaction.

MORPH's sparse-weight system: score at tile (16x16 saliency), execute at block
(128x128 BCSR via the vendored stk Triton backend), route at band (128 d_ff
neurons, ReMoE). See Ai-notes/06-09-2026/MORTAR/PLAN.md.
"""
