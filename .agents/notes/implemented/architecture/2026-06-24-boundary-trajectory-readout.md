# Agent Note: Boundary Trajectory Readout

Status: implemented

Origin: Ai-notes/06-24-2026/boundary-trajectory-readout/TECHNICAL_MEMO.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Boundary-Trajectory Tape: Decision Rule, Coupling, and Stability

Date: 2026-06-24

Runnable artifacts in this folder:

- `boundary_toy_sim.py`: numpy-only geometry, conditional-MI, and crossover probes.
- `stability_z3.py`: z3 proof script for a sufficient tape-feedback stability envelope.

## 0. Plain Problem

A causal LM already stores content in the current head-input latent `z_t`. Prior ablations say content retrieval, mean pooling, and compressed attention memory are mostly redundant. The only plausible new signal is not "what content was retrieved" but "how the state has been moving at boundary times."

So the problem is:

> Given boundary latents `zeta_k = z_{b_k}`, when is a tape of recent boundary states useful, which non-content reader should read it, what coupling loss makes the trajectory more forecastable without collapse, and where can the tape inject without destabilizing the looped core?

The answer is a decision rule over measurable quantities:

```text
MI = I(future residual ; old tape | local recency window)
phi_traj = MI(trajectory features ; future residual | local) / MI(old tape ; future residual | local)
L = R2_linear_lastK / R2_nonlinear_lastK on the same nonlocal residual target, clipped to [0, 1]
pi = pivot rate = Pr[curvature > calibrated threshold]
rho_core = measured loop-core spectral radius/order parameter
```

## 1. Operational Definitions

Boundary velocity and curvature:

```text
Delta_k = zeta_k - zeta_{k-1}
a_k = Delta_k - Delta_{k-1}
kappa_k = ||a_k|| / (||Delta_k|| + ||Delta_{k-1}|| + eps)
```

For noisy boundary latents, use a short-stride turning estimate:

```text
u_k = zeta_k - zeta_{k-s}
v_k = zeta_{k+s} - zeta_k
kappa_k^turn = 0.5 * (1 - <u_k, v_k>/(||u_k||||v_k|| + eps))
```

Calibrate the pivot threshold `tau_kappa` from a smooth/null baseline, for example the 95th percentile under linear flow or within-document non-pivot spans:

```text
pi = mean_k[ kappa_k > tau_kappa ].
```

Intrinsic dimension:

```text
C = Cov(zeta)
lambda_1 >= ... >= lambda_d
d_PR = (sum_i lambda_i)^2 / sum_i lambda_i^2
d_90 = min m such that sum_{i<=m} lambda_i / sum_i lambda_i >= 0.90
```

The forecast target must not be the near-future latent itself. That is the P4 trap. Define a local window `W_k = zeta_{k-w+1:k}` and an old tape `T_k = zeta_{k-K+1:k-w}`. Let

```text
F_{k,h} = zeta_{k+h}, h > w
B_h(W_k) = best local-window predictor of F_{k,h}
Y_{k,h} = F_{k,h} - B_h(W_k)
```

All readout probes and `L` below should be computed on `Y_{k,h}`, not on `zeta_{k+1}`.

## 2. Q1/Q2: Geometry and Information

Predictive probes:

```text
R2_z          = R2(zeta_{k+1} ~ zeta_k)
R2_z_plus_v   = R2(zeta_{k+1} ~ zeta_k + Delta_k)
R2_linear_K   = R2(Y_{k,h} ~ linear(last K zetas))
R2_nonlinear_K= R2(Y_{k,h} ~ nonlinear order-aware reader(last K zetas))
L             = min(1, max(0,R2_linear_K)/max(eps,max(0,R2_nonlinear_K)))
```

Conditional MI is estimated by a predictive-probe Gaussian lower bound:

```text
I(F; T | W) = H(F | W) - H(F | W,T)
           ~= 0.5 log det Sigma(F | W) / det Sigma(F | W,T)
```

In code this is measured per latent dimension. An InfoNCE version is equivalent for decision purposes:

```text
I(F;T|W) >= log N - L_NCE(s_theta(F,T,W)).
```

Decompose the old tape:

```text
G_k = trajectory features from T_k
    = [last old zeta, mean velocity, last velocity, curvature stats, low-rank trajectory coeffs]
C_k = T_k - Proj_G(T_k)
```

Then estimate:

```text
MI_tape = I(Y; T | W)
MI_traj = I(Y; G | W)
MI_resid = I(Y; C | W,G) or I(Y; C | W)
phi_traj = MI_traj / max(MI_tape, eps)
```

Interpretation:

- If `MI_tape ~= 0`, the tape idea is dead for that layer/horizon.
- If `MI_traj` explains most of `MI_tape`, the useful signal is trajectory geometry.
- If `MI_resid` dominates, the tape is carrying content-like residuals and will likely regress to the redundant retrieval family P1.

### Toy Probe Results

Command:

```bash
python Ai-notes/06-24-2026/boundary-trajectory-readout/boundary_toy_sim.py
```

Exit code: 0.

Important rows:

```text
kind         d_PR   pi      one-step L  R2_cond_linear  R2_cond_nonlin  L_cond  MI_tape  MI_traj
linear       2.10   0.036   1.0009      0.7641          0.7380          1.000   0.2585   0.2613
piecewise    2.87   0.160   1.0006      0.3786          0.3927          0.964   0.0701   0.0745
piecewise    2.78   0.189   1.0002      0.0267          0.0802          0.333  -0.0058  -0.0017
nonlinear    3.88   0.168   1.0003      0.5347          0.5605          0.954   0.1269   0.1277
random walk  2.16   0.878   1.2189     -0.0340         -0.0125          0.000  -0.0150  -0.0095
```

The one-step latent task is saturated, exactly as P4 predicts. The nonlocal residual task is the useful diagnostic.

Crossover command:

```bash
python Ai-notes/06-24-2026/boundary-trajectory-readout/boundary_toy_sim.py --crossover
```

Exit code: 0. In a toy where switch cues are nonlinear but observable from the trajectory, M4's conditional residual advantage is positive across the grid:

```text
pivot_p switch_mag pi     L_cond  M4_cond_advantage
0.02    0.4        0.108  0.451   0.145
0.08    0.8        0.175  0.222   0.195
0.18    0.8        0.189  0.177   0.193
0.18    1.6        0.193  0.000   0.111
```

This is not evidence about the real LM. It validates the decision geometry: M4 wins when the nonlinear/order-aware switch information is observable and nonlocal.

## 3. Q3: Optimal Readout Derivation

### 3.1 Linear-Gaussian Dynamics

Assume hidden state `s_k`:

```text
s_{k+1} = A s_k + eta_k,        eta_k ~ N(0,Q)
zeta_k  = H s_k + eps_k,        eps_k ~ N(0,R)
```

The MMSE readout is the conditional mean:

```text
m_{k|k} = E[s_k | zeta_{1:k}]
P_{k|k} = Cov[s_k | zeta_{1:k}]

m_{k+1|k} = A m_{k|k}
zeta_hat_{k+1|k} = H A m_{k|k}
```

The Kalman recursions give:

```text
m_{k|k-1} = A m_{k-1|k-1}
P_{k|k-1} = A P_{k-1|k-1} A^T + Q
K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^-1
m_{k|k} = m_{k|k-1} + K_k (zeta_k - H m_{k|k-1})
```

For a constant-velocity state

```text
s_k = [position_k, velocity_k]
A = [[I, I],
     [0, beta I]]
H = [I, 0]
```

and high observation SNR, the steady-state filter reduces to:

```text
zeta_hat_{k+1} ~= zeta_k + alpha * (zeta_k - zeta_{k-1})
```

This is M2. If the absolute position term is redundant/content-like and only motion carries incremental information, the MMSE innovation component reduces to a smoothed velocity:

```text
vbar_k = beta vbar_{k-1} + (1-beta)(zeta_k-zeta_{k-1})
```

This is M3.

Therefore:

```text
Linear-Gaussian + low pi + L ~= 1 + MI > 0  => M2/M3 are optimal.
```

### 3.2 Piecewise-Linear Dynamics With Switches

Let regime `q_k` select dynamics:

```text
s_{k+1} = A_{q_k} s_k + eta_k
Pr(q changes at k) = pi
```

A global linear predictor uses:

```text
Abar = E[A_q]
s_hat^lin_{k+1} = Abar s_k
```

An order-aware reader estimates `q_k` or pivot confidence `u_k` from the ordered tape:

```text
s_hat^M4_{k+1} = E[A_q s_k | T_k]
```

Assume for the moment M4 infers the regime perfectly. The excess MSE of the global linear prior is:

```text
E_lin - E_M4
= E ||(A_q - Abar)s_k||^2
= Tr( Sigma_s * E[(A_q-Abar)^T(A_q-Abar)] )
```

For two regimes with `Pr(q=1)=pi`, `Delta A = A_1 - A_0`:

```text
E_lin - E_M4 = pi(1-pi) * Tr(Sigma_s DeltaA^T DeltaA).
```

For semantic pivots as velocity rotations, linear extrapolation error at a pivot is:

```text
e_pivot = (R_theta - I) v_k
E||e_pivot||^2 = E[ 4 sin^2(theta/2) ||v_k||^2 ].
```

So the average M4 advantage is approximately:

```text
Delta_M4 ~= pi * delta_switch^2 * p_obs - C_est
```

where:

```text
delta_switch^2 = E[4 sin^2(theta/2)||v||^2]
p_obs = switch observability from ordered tape
C_est = finite-sample/regularization cost of the larger reader
```

If switches are random and unobservable, `p_obs ~= 0`, so M4 has no principled advantage. It can only overfit.

### 3.3 Crossover Rule

Let:

```text
DeltaR2 = R2_nonlinear_K - R2_linear_K = R2_nonlinear_K * (1 - L)
```

M4 is strictly better when:

```text
MI_tape > tau_MI
phi_traj >= tau_traj
DeltaR2 > tau_R2
pi * delta_switch^2 * p_obs > C_est
```

Since `delta_switch^2` and `p_obs` are hard to know directly, use the measured residual-probe quantities:

```text
M*(L, pi, MI, rho_core):

1. If MI_tape <= tau_MI:
       no production tape. Keep only diagnostics/control.

2. If MI_tape > tau_MI but phi_traj < tau_traj:
       no production trajectory tape. The useful signal is content-like; use Mneg only as a danger control.

3. If MI_tape > tau_MI, phi_traj >= tau_traj, L >= 1 - tau_L, and pi <= tau_pi:
       use M2 if absolute boundary position survives the raw-content/position control;
       otherwise use M3 velocity-only.

4. If MI_tape > tau_MI, phi_traj >= tau_traj, and (L < 1 - tau_L or pi > tau_pi):
       use M4, but only if the nonlinear residual-probe advantage
       R2_nonlinear_K - R2_linear_K >= tau_R2 on held-out blocked splits.

5. Never use Mneg as the production readout. It is the owned negative.
```

Starting thresholds for a first probe:

```text
tau_MI   = 0.01 nats/dim beyond local window
tau_traj = 0.60
tau_L    = 0.05
tau_pi   = calibrated smooth-null p95 rate, often around 0.10 to 0.20 in the toy
tau_R2   = 0.02 absolute conditional R2
```

Recommendation from priors plus derivation:

> Default to M4 only after the probes show nonlocal trajectory MI and a held-out nonlinear residual advantage. Otherwise use M2/M3. If MI is null, build no production tape.

## 4. Q4: Coupling Objective and Collapse

Let `Delta_k = zeta_k - zeta_{k-1}`.

### 4.1 Curvature Minimization

```text
L_curv = sum_k ||zeta_{k+1} - 2 zeta_k + zeta_{k-1}||^2
```

Global minimizers satisfy:

```text
zeta_{k+1} - 2 zeta_k + zeta_{k-1} = 0
=> zeta_k = a + k b
```

for each coordinate. Constant collapse is the special case `b=0`; low-rank line collapse is the general case. This explains P3: curvature minimization improves forecastability by flattening the path, but it can remove state variation the LM head/loop needs.

### 4.2 Direction-Only STP

```text
L_dir = sum_k 1 - cos(Delta_k, Delta_{k-1})
```

Ignoring zero-vector implementation details, global minimizers satisfy:

```text
Delta_k = c_k u,   c_k > 0
```

for one fixed direction `u`. Thus `zeta_k` lies on a line with arbitrary speed. With epsilon-normalized cosines, `Delta_k = 0` often becomes a silent numerical minimizer or stationary point. So STP also admits constant/low-rank collapse.

### 4.3 Scale-Invariant Forecast Residual

Define a nonlocal residual target:

```text
Y_{k,h} = zeta_{k+h} - B_h(W_k),     h > w
```

and a trajectory reader `q_theta(T_k)`. A scale-invariant residual objective is:

```text
L_sifr = || norm(q_theta(T_k)) - stopgrad(norm(Y_{k,h})) ||^2
```

By itself this still admits collapse because `Y_{k,h}` is a function of trainable latents. If all boundary latents become constant, the residual is zero/undefined and the normalized implementation can hide the collapse.

### 4.4 Variance-Preserving Coupling

Let `r_k` be the trajectory representation being coupled, for example whitened residual innovations or boundary deltas. Batch covariance:

```text
C = Cov(r_k)
```

A hard anti-collapse condition is:

```text
lambda_min(C) >= nu > 0
```

on the active subspace. A VICReg/SIGReg-style soft version:

```text
L_var = sum_j max(0, gamma - std(r_:,j))^2
L_cov = sum_{i != j} C_ij^2
```

A constant solution has `C=0`, so it violates `L_var` in every active dimension. With a penalty rather than a hard constraint, collapse is not a global minimizer if:

```text
lambda_var * r * gamma^2 > L_feasible_noncollapsed - L_forecast_collapse
```

The clean version is to enforce whitening/flooring as a constraint or use a large enough adaptive penalty that the variance floor is never violated in accepted runs.

### Recommended Coupling

Use a variance-preserving, scale-invariant, nonlocal residual forecast loss:

```text
L_couple =
    || norm(q_theta(G_k)) - stopgrad(norm(Whiten(Y_{k,h}))) ||^2
  + lambda_var * L_var(r_k)
  + lambda_cov * L_cov(r_k)
```

where:

```text
h > local recency horizon
G_k = trajectory geometry from tape, not content attention
Y_{k,h} = future latent residual after subtracting the local-window baseline
r_k = boundary delta/innovation representation used by the reader
```

This is compatible with M2/M3 by making `q_theta` linear, and compatible with M4 by making `q_theta` a small causal Conv1D/GRU/SSM/MLP-over-flattened trajectory reader.

Do not use pure curvature minimization or pure STP as the production coupling. They are useful diagnostics but have the wrong global minimizers.

## 5. Q5: Stability Envelope and z3 Proof

Linearize the tape-augmented per-token recurrence:

```text
delta z_{n+1}
  = A delta z_n + sum_{j=0}^{K-1} B_j delta z_{n-j}

||A|| <= rho_core
sum_j ||B_j|| <= gamma
gamma = g * ||W|| * ||R||
```

For an eigen-root with magnitude `r = |lambda|`:

```text
r <= rho_core + gamma * sum_{j=0}^{K-1} r^{-j}.
```

If we want no root outside `rho_star`, it is sufficient that:

```text
rho_core + gamma * S_K(rho_star) <= rho_star
S_K(rho_star) = sum_{j=0}^{K-1} rho_star^{-j}
```

because any root with `r >= rho_star` would imply:

```text
r <= rho_core + gamma * S_K(rho_star) <= rho_star
```

a contradiction for `r > rho_star`.

Thus:

```text
g <= (rho_star - rho_core) / (||W||||R|| S_K(rho_star)).
```

If injection happens inside each loop iteration rather than after the loop, the effective tape gain is amplified:

```text
H_T = sum_{m=0}^{T-1} rho_core^m
g <= (rho_star - rho_token) / (||W||||R|| H_T S_K(rho_star))
```

where `rho_token` is the measured token-level loop Jacobian radius after `T` iterations. This is why core injection is expensive.

### z3 Script

Full script: `stability_z3.py`.

Core proof encoded:

```python
assumptions = And(
    rho_star > 0,
    rho_core >= 0,
    gamma >= 0,
    s_k >= 1,
    rho_core < rho_star,
    gamma * s_k <= rho_star - rho_core,
    bound == rho_core + gamma * s_k,
)
claim = bound <= rho_star
solver.add(assumptions)
solver.add(Not(claim))
```

Command:

```bash
/tmp/morph-z3-venv/bin/python Ai-notes/06-24-2026/boundary-trajectory-readout/stability_z3.py --rho-core 0.92 --rho-star 0.98 --k 6 --wr 1.0
```

Exit code: 0.

Output:

```text
PROVED: assumptions imply rho_core + gamma*S_K <= rho_star
S_K=6.314581
gamma_max=0.009502
g_max=0.009502 for ||W||||R||=1.000000
gate_init_recommendation=0.000950
```

Safe gate examples for `K=6`, `rho_star=0.98`, `||W||||R||=1`:

```text
rho_core  gamma_max  gate_init = 0.1 gamma_max
0.85      0.020587   0.002059
0.92      0.009502   0.000950
0.97      0.001584   0.000158
```

Conclusion:

> Do not inject the tape into the looped core by default. The safe gate is so small near `rho_core ~= 1` that core injection is either functionally weak or destabilizing. Inject into CODA/readout after the loop unless a measured spectral-margin test proves the core envelope with the actual `T`, `K`, `||W||||R||`, and gate derivative terms.

For CODA/readout-only injection, the loop recurrent Jacobian remains block triangular with the original core block. The tape can affect logits/readout but does not add a second hidden-state recurrence in the loop. This is the stable default.

## 6. Synthesis

### 6.1 Recommended M*

Use this decision rule:

```text
Measure on held-out boundary latents:
  MI_tape, MI_traj, MI_resid, phi_traj, L_cond, pi, rho_core.

If MI_tape <= 0.01 nats/dim:
  no production tape.

Else if phi_traj < 0.60 or MI_resid >= MI_traj:
  no production trajectory tape; Mneg/content path only as danger control.

Else if L_cond >= 0.95 and pi <= tau_pi:
  use M2 if position+velocity beats position/raw-content controls;
  otherwise use M3 velocity-only.

Else if L_cond < 0.95 or pi > tau_pi:
  use M4 only if nonlinear residual-probe advantage >= 0.02 absolute R2 on blocked held-out splits.

Injection site:
  CODA/readout by default.
  Core injection only if g <= (rho_star-rho_core)/(||W||||R||S_K)
  after including loop-iteration amplification.
```

One principled recommendation:

> Build M4 as the default experimental production candidate, but gate its use by the decision rule above. If the probes do not show nonlocal trajectory MI and nonlinear/order-aware advantage, flip down to M2/M3 or no tape. Do not build content-addressed tape except as Mneg control.

### 6.2 Recommended Coupling

Use:

```text
variance-preserving scale-invariant forecast residual
on future residuals beyond the local recency window
from trajectory geometry features
```

Do not use pure curvature or pure direction smoothing as production coupling.

### 6.3 Experimental Arms

Minimal arms:

```text
A0 baseline: no tape, no coupling
A1 Mneg: raw last-K content-addressed tape, no coupling
A2 selected readout M* tape, no coupling
A3 selected readout M* tape, recommended coupling
A4 selected readout M* tape, bad coupling control: pure curvature or STP
```

The single comparison that proves "coupling mattered" rather than "we added a tape" is:

```text
A2 selected readout M* tape, no coupling
vs
A3 selected readout M* tape, recommended coupling
```

Everything else is control context.

### 6.4 Kill Metrics

Primary:

```text
zero-the-tape sensitivity:
  run with tape active, then zero tape/gate at eval;
  require a measurable drop/change on long-range stateful tasks or latent/logit KL.

gate utilization:
  mean gate, entropy, fraction gate > epsilon, per-layer/per-step histogram;
  lazy-gate collapse by 2k-3k steps is a kill condition.

stability:
  measured rho_core and rho_aug via power iteration/JVP;
  abort core injection if rho_aug > rho_star or if gate violates envelope.

owned negatives:
  local recency predictor
  Mneg content-addressed tape
  position-only and velocity-only probes
```

Perplexity is secondary/noisy for this mechanism.

### 6.5 Falsifiable Predictions Before Full Training

1. Boundary latent probes will show one-step `zeta_{k+1}` prediction saturated, but nonlocal residual probes will separate regimes. If `MI_tape <= 0.01 nats/dim`, the tape will be behaviorally dead.

2. If `L_cond < 0.95` and `pi` is above the smooth-null pivot rate, M4 will beat M2/M3 by at least `0.02` absolute conditional residual `R2` before full training. If it does not, use M2/M3.

3. For `rho_core >= 0.92`, safe core-injection gates will be around `1e-3` to `1e-2` times `1/(||W||||R||)` for `K=6`. A normal-sized core gate will either violate the envelope or need spectral normalization so strong it becomes functionally weak. CODA/readout injection should not move `rho_core`.

## 7. What To Build

Build a CODA/readout-only M4 trajectory adapter first: a small causal Conv1D/GRU/SSM/MLP-over-flattened reader over the last `K` boundary latents that emits `(prior p_k, velocity v_k, curvature c_k, confidence u_k)`, injects `g * W(p_k + v_k)` after the loop, and uses `c_k,u_k` only for the gate. Train it with the variance-preserving scale-invariant forecast-residual coupling at horizon `h > local_window`, with full gate-utilization logs and zero-tape sensitivity. Run M2/M3 and Mneg controls. Promote M4 only if the preflight probes satisfy `MI_tape > tau_MI`, `phi_traj >= tau_traj`, and `R2_nonlinear - R2_linear >= tau_R2`; otherwise flip to M2/M3 or no tape.

## 8. Verification Performed

Commands run:

```bash
python -m py_compile Ai-notes/06-24-2026/boundary-trajectory-readout/boundary_toy_sim.py Ai-notes/06-24-2026/boundary-trajectory-readout/stability_z3.py
python Ai-notes/06-24-2026/boundary-trajectory-readout/boundary_toy_sim.py
python Ai-notes/06-24-2026/boundary-trajectory-readout/boundary_toy_sim.py --crossover
/tmp/morph-z3-venv/bin/python Ai-notes/06-24-2026/boundary-trajectory-readout/stability_z3.py --rho-core 0.85 --rho-star 0.98 --k 6 --wr 1.0
/tmp/morph-z3-venv/bin/python Ai-notes/06-24-2026/boundary-trajectory-readout/stability_z3.py --rho-core 0.92 --rho-star 0.98 --k 6 --wr 1.0
/tmp/morph-z3-venv/bin/python Ai-notes/06-24-2026/boundary-trajectory-readout/stability_z3.py --rho-core 0.97 --rho-star 0.98 --k 6 --wr 1.0
```

All listed commands exited 0 in this run.

Not verified:

- No real MORPH/LM boundary latents were extracted.
- No actual Hydra/wandb training arm was launched.
- No empirical `rho_core` was measured on the live model.
- The z3 proof is a sufficient norm-bound proof, not a tight spectral-radius characterization.
