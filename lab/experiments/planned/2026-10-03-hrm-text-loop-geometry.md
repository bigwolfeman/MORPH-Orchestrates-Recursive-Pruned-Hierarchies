# Planned: does HRM-Text's two-level loop rotate at fixed scale like Huginn's?

Status: planned

Date: 2026-10-03 01:52 CDT. Predictions frozen before the probe exists and before I read HRM-Text's code.

## Question

Huginn's loop earns depth by rotating its state at a fixed per-token scale, with steps
99.6 % to 99.9 % perpendicular to a small shared bias direction and diversity (PR) tripling
in four iterations ([filing](../successes/2026-10-02-huginn-loop-geometry.md)). Wolfe
(2026-10-03): "I think we should see how that one behaves too." HRM-Text-1B
(`sapientinc/HRM-Text-1B`) is a second looped language model with a different design: two
modules, a slow H state and a fast L state, iterated H_cycles x (L_cycles + 1) times with
additive state injection. Does a two-level loop behave the same way, and does its depth
earn on our web text at all?

## Hypothesis

The geometry is a property of an earning loop, not of Huginn alone: HRM's states also move
in direction at a stable scale, mostly perpendicular to their shared direction, with
diversity growing in the first cycles. The H state changes slowly across H cycles and the
L state quickly within one.

## Predictions (mine, orchestrator)

Same 64 rows x 1024 tokens as the Huginn probe; readings per H cycle and, inside one H
cycle, per L step; v is each state's unit mean over tokens and steps.

- **R-1**: HRM-Text earns depth on our rows: CE at its trained cycle count is below CE at
  one H cycle by more than 0.1 nats. 60 % (it was trained on structured data, PrefixLM,
  not on web text).
- **R-2**: each state's norm at the last step is within 1.5x of its norm after the first
  H cycle. 75 % (I expect a norm on the state path, as in Huginn, but have not checked).
- **R-3**: from the second H cycle on, under 30 % of the H state's mean squared step lies
  along v_H. 65 %.
- **R-4**: the H state's relative step per H cycle is smaller than the L state's relative
  step per L step. 70 %.
- **R-5**: the H state's PR grows by more than 30 % from the first to the last H cycle.
  50 %.
- **R-6**: removing a random direction from the final H state costs under 0.01 nats, and
  removing v_H costs more than 0.05 nats. 60 %.

## Method

Download `sapientinc/HRM-Text-1B` into the local HF cache. Build
`lab/hrm/hrm_text_loop_geometry.py` on the Huginn probe's pattern: capture both states
after every step without changing the computation (faithfulness: logits equal to the
model's own forward), the same readings, and the same three interventions (rescale, remove
v, remove a random direction) on the final H and L states, paired row-bootstrap CIs. If
HRM-Text needs a prefix or condition token, use its documented default for plain text and
record it here as a Method amendment before the run.
