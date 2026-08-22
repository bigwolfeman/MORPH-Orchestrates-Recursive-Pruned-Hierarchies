"""Gate startup instruments — bias seating and the travel audit.

docs/tul-gate-spec.md §10. Every function here exists because the predecessor
(``00DeepNet/coconut``) lost a whole ladder to a gate that never moved: its bias went
−2.00000 → −2.00071 against a required travel of 1.88, 57× short of what the step budget
could deliver, and nothing in the run said so. The loss curve looked fine, because a
constant predictor at the corpus base rate IS a low-loss solution.

Two instruments, in the order they run:

1. :func:`seat_gate_bias` — start the bias AT the base rate instead of making it travel
   there. One batch of arithmetic; removes the failure mode rather than detecting it.
2. :func:`audit_gate_travel` — REFUSE to start a run whose gate provably cannot move:
   parameters missing from the optimizer, a zero learning rate, or a bias budget smaller
   than the distance the bias must cover.

and one that runs later:

3. :func:`assert_gate_is_alive` — a few thousand steps in, a gate direction still at its
   zero init is a frozen gate. Fail then, not at hour three.
"""

from __future__ import annotations

import math

import torch

__all__ = ["assert_gate_is_alive", "audit_gate_travel", "seat_gate_bias"]


def _logit(q: float) -> float:
    q = min(max(q, 1e-4), 1.0 - 1e-4)
    return math.log(q / (1.0 - q))


@torch.no_grad()
def seat_gate_bias(gate, layout) -> dict:
    """Seat ``gate.b`` at the corpus base rate and return the target statistics.

    Uses ONE real batch's ``span_len``, not a hardcoded constant: the base rate depends on
    the tokenizer, ``min_span`` and ``span_cap``, and a number that is not measured is a
    number that goes stale silently.
    """
    if layout.span_len is None:
        raise RuntimeError("seat_gate_bias needs a layout carrying span_len (gate §3.3)")
    span, valid = layout.span_len, layout.slot_valid
    sel = valid & (span > 0)
    if not bool(sel.any()):
        raise ValueError("seat_gate_bias: the batch has no valid slot")
    b = gate.seat_bias(span, valid)
    t = span[sel].float() / gate.gate.k_max
    q10, q50, q90 = (float(torch.quantile(t, q)) for q in (0.1, 0.5, 0.9))
    return {
        "gate_seated_bias": b,
        "gate_target_mean": float(t.mean()),
        "gate_target_q10": q10,
        "gate_target_q50": q50,
        "gate_target_q90": q90,
        "gate_span_mean": float(span[sel].float().mean()),
        "gate_cap_frac": float((span[sel] == gate.gate.k_max).float().mean()),
    }


@torch.no_grad()
def audit_gate_travel(gate, optimizer, stats: dict, total_steps: int, z_norm: float) -> dict:
    """Raise unless the gate can provably reach the targets within the step budget.

    The budget model is AdamW's: the second-moment normalisation makes one step's update
    magnitude ≈ ``lr`` per component, so ``lr · steps`` is an UPPER bound on how far any
    single parameter can travel. A necessary condition, which is what a pre-flight audit
    needs — it never passes a gate that cannot move, and it never blocks one that can.

    Two travels are checked:

    * the BIAS must cover ``|logit(median target) − b_init|``. After :func:`seat_gate_bias`
      this is ~0 and the check passes by construction; it fires loudly if seating is ever
      removed, which is exactly the predecessor's configuration.
    * the DIRECTION must cover the SPREAD, ``|logit(q90) − logit(q10)|``. Its budget is
      ``lr · steps · ‖z‖``, the logit swing available if every component of ``w`` moves
      coherently against a readout input of the measured norm.

    Args:
        z_norm: the measured L2 norm of the gate's normalised input on the seating batch.
                Measured, not assumed — RMSNorm's learnable scale drifts.
    """
    named = {f"tul_gate.{n}": p for n, p in gate.named_parameters()}
    lrs, wds = {}, {}
    for name, p in named.items():
        if not p.requires_grad:
            raise RuntimeError(
                f"gate audit: {name} has requires_grad=False — the gate cannot train. "
                f"(docs/tul-gate-spec.md §10)"
            )
        hit = [g for g in optimizer.param_groups if any(q is p for q in g["params"])]
        if len(hit) != 1:
            raise RuntimeError(
                f"gate audit: {name} appears in {len(hit)} optimizer param groups, "
                f"expected exactly 1. A gate outside the optimizer is the deadest "
                f"possible gate and the loss curve will not show it."
            )
        lrs[name] = float(hit[0]["lr"])
        wds[name] = float(hit[0].get("weight_decay", 0.0))
        if lrs[name] <= 0.0:
            raise RuntimeError(f"gate audit: {name} is in a group with lr={lrs[name]}")

    lr_b, lr_w = lrs["tul_gate.b"], lrs["tul_gate.w"]
    need_bias = abs(_logit(stats["gate_target_q50"]) - float(gate.b.item()))
    need_spread = abs(_logit(stats["gate_target_q90"]) - _logit(stats["gate_target_q10"]))
    budget_bias = lr_b * total_steps
    budget_spread = lr_w * total_steps * z_norm
    out = {
        "gate_audit_need_bias": need_bias,
        "gate_audit_budget_bias": budget_bias,
        "gate_audit_need_spread": need_spread,
        "gate_audit_budget_spread": budget_spread,
        "gate_audit_z_norm": z_norm,
        "gate_audit_lr": lr_w,
        "gate_audit_wd": wds["tul_gate.w"],
    }
    fail = []
    if budget_bias < need_bias:
        fail.append(f"bias needs {need_bias:.3f} logits, budget lr·steps = {budget_bias:.3f}")
    if budget_spread < need_spread:
        fail.append(
            f"direction needs {need_spread:.3f} logits of spread, budget "
            f"lr·steps·‖z‖ = {budget_spread:.3f}"
        )
    if fail:
        raise RuntimeError(
            "gate audit REFUSED the run (docs/tul-gate-spec.md §10): "
            + "; ".join(fail)
            + ". Raise tul.gate_lr_mult / training.lr, or seat the bias. This is the check "
            "that the predecessor did not have when its gate travelled 0.04 % of what "
            "it needed and the ladder was scored anyway."
        )
    print(
        f"  [gate-audit] PASS  bias need={need_bias:.3f} ≤ {budget_bias:.3f} | "
        f"spread need={need_spread:.3f} ≤ {budget_spread:.3f} (‖z‖={z_norm:.1f}) | "
        f"lr={lr_w:g} wd={wds['tul_gate.w']:g} | seated b={float(gate.b.item()):+.4f} "
        f"target q10/q50/q90={stats['gate_target_q10']:.3f}/"
        f"{stats['gate_target_q50']:.3f}/{stats['gate_target_q90']:.3f} "
        f"span_mean={stats['gate_span_mean']:.1f} cap_frac={stats['gate_cap_frac']:.3f}",
        flush=True,
    )
    return out


@torch.no_grad()
def assert_gate_is_alive(gate, step: int, tol: float = 1e-6) -> None:
    """Fail the run if the gate direction is still at its zero init (§10).

    ``w`` starts at exactly 0 and receives a gradient on every step, so a norm still at
    the floor after thousands of steps means the parameter is frozen — the gate is
    emitting a constant whatever its loss says. Better to stop here than to score an arm
    whose mechanism never engaged.
    """
    n = float(gate.w.detach().norm())
    if n < tol:
        raise RuntimeError(
            f"gate is DEAD at step {step}: ‖w‖ = {n:.3e} is still its zero init, so the "
            f"gate emits a constant for every slot. Check that tul_gate parameters reach "
            f"the optimizer and that gate_lambda > 0. (docs/tul-gate-spec.md §10)"
        )
    print(
        f"  [gate-audit] alive at step {step}: ‖w‖={n:.4f} b={float(gate.b.item()):+.4f}",
        flush=True,
    )
