# MORPH Hyper-Connection residual vs JPmHC: audit

Context: the Parcae testbed panel (parcae branch lxtul-testbed, docs/experiments/failures/2026-10-06-parcae-lxtul-5k-panel.md) found the LXTUL slot loop earns K1-K6 +0.0585 on a single-stream backbone vs +0.0215 on MORPH. This audit asks whether MORPH's HC departs from the paper in a way that explains the slot-shared constant in the stream differences.

Date: 2026-10-06. Repo: MORPH master 3f9aae2 (read only, no edits).
Auditor: read-only subagent. No GPU. Two small CPU scripts (niced, one core): `lab/divergence/hc_jpmhc_audit_check.py` (math checks) and `lab/divergence/hc_jpmhc_audit_ckpt.py` (reads the static biases
from one checkpoint).

## Paper used

- Sengupta, Wang, Brunswic (LLM Suite Team, JP Morgan Chase), "JPmHC: Dynamical Isometry via
  Orthogonal Hyper-Connections", arXiv 2602.18308, Feb 2026.
- Local copy: `docs/references/residual-streams/jpmhc/jpmhc.pdf` (33 pages) and the
  pymupdf4llm text `docs/references/residual-streams/jpmhc/jpmhc.md`. The markdown drops
  the equations ("picture omitted"), so I read every equation below from the PDF with
  `pdftotext -layout` (pages 2-3, 6-8, 31-33).
- Paper equations used: (2) block form, (11)/(31)-(32) iterative Cayley, (12)/(35) fused
  projection, (13)/(36)-(38) constraint maps, (14)/(39)-(43) forward, Fig. 2 (block diagram),
  Table 1 (variant config), Table 9 (parameter count), §7.5 (limitations), App. I.

## Legend

- MATCH: same math as the paper.
- DEVIATES: different math. I say whether the repo documents it.
- MISSING: the paper has it, MORPH does not.
- UNSPECIFIED: the paper does not say; MORPH had to choose.
- [read] = I verified by reading code or paper. [ran] = I ran a check (numbers below).
  [inferred] = my reasoning, not measured.

## The paper's method in one block (from the PDF)

```
[H~pre | H~post | H~res] = W_fused · LayerNorm(x_flat),  W_fused ∈ R^{3n² × np}   (12)/(35)
Hpre  = softmax(H~pre / τ, dim=-1)        row-stochastic                         (36)
Hpost = softmax(H~post / τ, dim=-2)       column-stochastic                      (37)
Hres  = Cay_s(H~res):  W = H~ − H~ᵀ ; Y0 = I + αW ; Y_{i+1} = I + (α/2) W (I + Y_i), s=2, α=0.1
x_pre = Hpre · x_streams ; x_in = mean_stream(x_pre) ; y = F(x_in)               (39)-(41)
y_streams = s · Hpost,sum ⊙ y            (s is never defined in the paper)       (42)
x_out = Hres · x_streams + y_streams     ( = Hres x + Hpost (y ⊗ 1_n), eq. 14 )  (43)
Fig. 2: "both paths merge at the + node before Layer Norm"  (post-norm, TRM style)
Table 1 (Cayley row): s=2, α=0.1, LayerNorm, softmax.
Table 9: Cayley params = 48·nd + LayerNorm = 102,400 at nd=2048  -> W_fused has NO bias.
```

The paper does NOT specify: the W_fused init, how the n streams are created at the network
input, how they are reduced at the output, the value of τ, or what "s" in (42) is.

## Component table

| # | Component | Paper | MORPH (file:line) | Verdict |
|---|---|---|---|---|
| 1 | Coefficient input norm | LayerNorm(x_flat) with affine (eq. 12/35; Table 1, Table 9) | RMSNorm of the flattened n·C carrier, no affine, applied as `(x·Wᵀ)/rms + b` (`hyper_connections.py:168-174`; kernel `fused_hyper_connection.py:366-395`, `1401-1419`) | DEVIATES. Documented only in code comments ("mHC §4.3.1" reorder, "bias-under-rms" fix). No note says "paper uses LayerNorm". Numerically small for the issue below (see F3) |
| 2 | Fused projection | one linear, 3n² outputs, no bias (Table 9 arithmetic: 48·2048 + 2·2048 = 102,400) | `nn.Linear(nd, 3n², bias=True)`, bias zero-init (`hyper_connections.py:139-141`). Bias is in the optimizer's no-decay group (`optimizer.py:26` "bias") | DEVIATES (extra free static bias, no weight decay). Paper's LN β gives an effective bias `W·β`, so capacity is close [inferred]. Not documented as a deviation. Trained core biases are ~0 (F4) |
| 3 | Dynamic vs static coefficients | purely dynamic (input-dependent), LN affine gives an implicit static part | dynamic + static bias | MATCH in effect. Measured: in the LXTUL 5k checkpoint every core HC bias has max abs 0.00-0.02, so the core routing is ~100 % dynamic [ran] |
| 4 | Nonlinearity | softmax with temperature τ for pre (dim -1) and post (dim -2); no tanh, no sigmoid (eq. 36-37; "softmax rather than sigmoid") | `softmax(pre/τ,-1)`, `softmax(post/τ,-2)` (`hyper_connections.py:177-178`; kernel `392-417`), τ = 1.0 (`base.yaml:270`) | MATCH. τ value UNSPECIFIED in paper |
| 5 | H_pre read | x_in = mean_i (Hpre x)_i  ⇒ weights = column means of Hpre (eq. 39-40) | `Hpre_cm = Hpre.mean(-2)`; `x_bar = Σ_j Hpre_cm_j h_j` (`hyper_connections.py:219-220`; kernel `386-400`, `461-467`) | MATCH. Note (both): only 3 of the 16 pre entries' DoF reach the output; x_bar is always a convex combination of streams |
| 6 | H_post write | y_streams_i = rowsum_i(Hpost) · y (eq. 14/42), column-stochastic ⇒ row sums ∈ [0, n], Σ_i = n | `Hpost_row = Hpost.sum(-1)`; out_i += Hpost_row_i · y (`hyper_connections.py:218`; kernel `407-426`, `966-969`) | MATCH, except the undefined scalar "s" in (42), which MORPH sets to 1 implicitly. The write into the stream DIFFERENCES is (Hpost_row_i − 1)·y, zero-sum by construction (F1) |
| 7 | H_res constraint | orthogonal O(n) (connected component, det +1) via the inverse-free Cayley iteration, s = 2, α = 0.1, generator B = (α/2)(H~ − H~ᵀ) (eq. 11, 30-32, Alg. 8) | EXACT Cayley `(I+B)(I−B)⁻¹`, same generator `B = ½α(A − Aᵀ)`, via the 4×4 Cayley-Hamilton closed form (`hyper_connections.py:48-97`; kernel `_cayley4_fwd` `144-181`, analytic VJP `183+`) | DEVIATES (better). It is the paper iteration's fixed point. Documented in the docstring and kernel comments. The cited root-cause note "Ai-notes 07-02-2026 Cayley-HC-Divergence" does NOT exist on disk (`ignore/Ai-notes/` has no 07-02-2026 dir); the change is commit 9fbe1dea "kernel bug and cayley fix" (2026-07-02) |
| 8 | Cayley iterations / α | s = 2 (paper says s=2 suffices, ‖YᵀY−I‖max < 1e-3), α = 0.1 | `hc_cayley_iters: 3` is IGNORED by the math (`hyper_connections.py:70-72`, kernel `428`), but the fused path RAISES/falls back unless iters == 3 (`fused_hyper_connection.py:1083`, `1159`, `1369-1371`). α = 0.1 (`base.yaml:272`) | α MATCH. Iteration count is dead config. `base.yaml:271` and `transformer.py:542` still say "s=2 paper, 3 = safety margin": stale. Setting it to 2 silently drops to the eager reference (speed only, same numbers) |
| 9 | Residual tau / scale on H_res | no τ on H_res | no τ on H_res (kernel `429`: "NO tau on res") | MATCH |
| 10 | Norm placement in the block | Fig. 2: merge at "+", then Layer Norm on x_out ∈ R^N (post-norm, as in TRM). Eq. 41 has F applied to x_in with no pre-norm | pre-norm: `norm_attn(x_bar)` / `norm_mlp(x_bar)` inside the sublayer (`mhc.py:494`, `530`); the carrier itself is never normalised by the block | DEVIATES. Not documented as a deviation anywhere I found (`hyper_connections.py`, `mhc.py`, `docs/references.md`). In MORPH nothing bounds the carrier magnitude inside the core except the LXTUL per-pass cell norm (row 14) |
| 11 | Init / identity at init | paper does not give an init. Its claim is that the mixer stays spectrum-preserving | W_fused std = 0.1/√(nd), bias 0 (`hyper_connections.py:140-141`); streams equal at expand. Docstring: "reduces to x + F(mean(x)) at step 0 (verified)" | UNSPECIFIED in paper. MORPH is APPROXIMATELY a plain residual, not exactly: at d=1024, H~ entries have std ~0.1, so `Hpost_row` deviates from 1 by up to 0.14, `Hpre_cm` from 0.25 by up to 0.037, `Hres − I` up to 0.05; output vs `x + F(mean x)` max abs diff 0.52 for unit-scale y; each HC call injects ~0.11 % of its output energy into stream differences at step 0 [ran]. The docstring's "verified" is an overstatement |
| 12 | Stream expansion at input | UNSPECIFIED (Fig. 2 just shows x ∈ R^N) | broadcast copy of the single-stream embedding into all n streams (`transformer.py:4377-4383`); every injection (x0, value embeds, bigram, DiagonalInjection source, `post_inject`) is a single-stream term broadcast into all streams (`mhc.py:116-118`, `hyper_connections.py:195-197`, kernel `951-971`) | UNSPECIFIED. Consequence: every external input lives in the stream-MEAN subspace. Only HC writes (rows 6-7) can put content into the stream DIFFERENCES |
| 13 | Contraction at output | UNSPECIFIED | uniform stream mean before `lm_mixer` + `final_norm` (`transformer.py:4805-4822`); `mux_readout: full` per-stream variant exists (`4824-4858`); the TUL `prefix_project` hands the coda all n streams | UNSPECIFIED. The mean is exactly blind to any zero-sum stream pattern. A difference-mode vector is readable downstream only through a later HC's coefficient path (W_fused reads the whole flattened carrier) or a non-uniform Hpre |
| 14 | Per-stream normalisation | none in the paper's HC module | LXTUL only: `tul.slot_cell_pass_norm: rms` = RMSNorm per stream over C after every slot-loop pass (`transformer.py:3894-3960`) | MORPH-only addition (not in paper). Rescales each stream; it preserves a ±v sign pattern, so it cannot remove the constant (F5) |
| 15 | Precision | bf16 mixed precision; not detailed for the mapping | mapping math fp32 in-kernel; the GEMV x·Wᵀ runs in bf16 with fp32 accumulate (fused: `torch.mm` in carrier dtype under autocast, `1094-1097`; eager `F.linear` under autocast `hyper_connections.py:173`). Docstring line 30 says "mappings are computed in fp32": true only after the GEMV. Training carrier is fp32 (probed in `.agents/notes/proposed/process/2026-08-21-throughput-budget-and-cloud-plan.md:105`; I did not re-probe on 3f9aae2) | MATCH in spirit. See F6 for a bf16-carrier hazard |
| 16 | Gradient path | standard autograd through s=2 iterations (Table 15 "Standard autograd"); contribution 4 claims implicit diff "e.g. Cayley" but App. H gives only Sinkhorn's | eager: autograd through the closed form; fused: analytic closed-form VJP (`_cayley4_vjp`), rms-path and bias grads explicit (`1116-1146`) | DEVIATES (closed-form VJP), consistent with row 7 |
| 17 | Fused vs eager numerics | n/a | post: fused accumulates `Hres·h + Hpost_row·y + term` in fp32, one rounding (`953-971`); eager reference does einsum + mul + add in carrier dtype (`1426-1433`). With an fp32 carrier both are fp32; the remaining difference is the bf16 GEMV in both paths. Ledger row `HC-Cayley-n4`: "similar loss trajectory" (not bit parity) | Not a paper item. I did not run the GPU parity test |
| 18 | Isometry claim | the free-probability analysis is for a FIXED mixer A_n (eq. 3, "independent of x, for theoretical analysis") | repo docs say "exact dynamical isometry (all singular values 1)" for the HC residual (`hyper_connections.py:7`, `references.md`, `base.yaml:265-267`, `transformer.py:536-539`) | Overclaim in MORPH's docs [inferred]. Hres is orthogonal as a 4×4 matrix at fixed coefficients. The block Jacobian also has ∂Hres/∂x·x, ∂Hpost/∂x·y and the write term, none of which is orthogonal. The repo's own ρ(J_core) work already measures the core as non-isometric |
| 19 | Number of HC modules per block | 2 (pre-attention, pre-FFN) | 2 (`mhc.py:370-371`, `mrr_attn` / `mrr_mlp`) | MATCH |
| 20 | Compute claim | §7.1: Cayley is 2.25× fewer FLOPs per module than Sinkhorn (Table 6: 256 vs 576) | `docs/references.md` and `base.yaml:265` say "5.2× cheaper H^res" | Doc error: I did not find 5.2× anywhere in the extracted paper text |

## Findings that bear on the slot-loop carrier constant

The probe result being explained (from
`.agents/notes/rejected/architecture/2026-10-05-slot-loop-carrier-constant.md` and
`lab/experiments/failures/2026-10-05-carrier-constant-twin.md`): after pass 1, 98-99.8 % of
the 4-stream carrier is ONE slot-shared vector with a per-stream sign pattern (+,-,-,+ or
+,-,+,-), which cancels in the stream mean; the core attention writes it through one stream
(Hpost row about [0, 0, 3.9, 0]); with a uniform attention HC the MLP's Cayley mixers rebuilt
a new one (pattern ++--). The same loop on a single-stream Parcae core earns 2.7x more.

Define the stream-difference subspace D = {h : Σ_i h_i = 0} and the mean subspace
M = {h_1 = ... = h_n}. The carrier constant lives in D.

### F1 (MATCH with the paper, most likely to matter): the paper's own H_post writes into D

With a column-stochastic Hpost, stream i receives `rowsum_i · y`. The part of the write that
lands in D is `(rowsum_i − 1) · y`, which sums to zero over i by construction. A row-sum
vector [0, 0, 3.9, 0] (allowed: row sums lie in [0, 4]) writes `[-1, -1, +2.9, -1] · y`,
which is a per-stream sign pattern with no mean. Every other input to the carrier (embedding
expand, injections, `post_inject`) is broadcast and lives in M (row 12). So the H_post write
and the H_res rotation are the ONLY two ways anything enters D, and the paper prescribes both.
This is not a MORPH bug. It is the paper's design. [read + inferred]

### F2 (MATCH with the paper, likely to matter): orthogonal H_res neither damps D nor keeps M and D apart

- A doubly-stochastic mixer (mHC) has H·1 = 1 and Hᵀ·1 = 1, so it maps M to M and D to D, and
  its non-Perron eigenvalues are inside the unit disk, so a vector parked in D decays over
  repeated application. JPmHC removes exactly this contraction (that is its selling point:
  "no eigenvalue contraction").
- An orthogonal H_res keeps ‖h‖ but does not keep the M/D split. Measured leak of the
  stream-mean direction into D, ‖(I − 11ᵀ/n) H 1/√n‖ per application: 0.023 at raw entry
  std 0.1, 0.22 at std 1, 0.89 at std 10 (doubly stochastic: exactly 0) [ran].
- So under JPmHC (paper and MORPH alike) a vector in D costs nothing to keep across passes,
  and an orthogonal mixer can rotate mean content into D. The uniform-HC twin result (the
  MLP Cayley mixers turned a stream-identical attention write into stream differences)
  matches this mechanism; that step was inferred from a decomposition in the filing, not
  causally tested, and I did not test it either.

### F3 (MATCH in effect, likely to matter): the coefficient path reads D, and a large D vector turns the dynamic HC into a fixed router

- W_fused reads the WHOLE flattened carrier (both paper and MORPH). The normaliser (paper
  LayerNorm, MORPH RMSNorm) does not remove a zero-sum pattern: LN subtracts the mean over all
  n·C entries, and for `v ⊗ s` with Σ_i s_i = 0 that mean is 0. So the RMS-vs-LN deviation
  (row 1) does NOT change this [inferred].
- If 98-99.8 % of the carrier energy is one fixed vector, the normalised coefficient input
  x̂ is that vector's direction plus a 4-14 % per-slot ripple (√0.002 to √0.02). The H
  logits become `64 · W · v̂` (‖x̂‖ = √(n·C) = 64 at C = 1024) plus a small per-slot part:
  a large CONSTANT shared by every slot. Measured trained core row norms ‖W_row‖ are
  0.11-0.88 [ran], so aligned logits of 7-55 are reachable, enough to saturate the column
  softmax and give row sums like [0, 0, 3.9, 0]. The checkpoint's core biases are ~0 [ran],
  so this routing is coming through the dynamic path, not a learned bias.
- That closes a loop [inferred, not measured]: a D-vector in the carrier makes the HC
  mappings constant and saturated, a saturated Hpost writes the attention output into D
  (F1), and an orthogonal Hres keeps it (F2). The paper's design allows this feedback; MORPH
  adds nothing that breaks it.

### F4 (DEVIATION, low relevance as measured): free, undecayed bias on W_fused

The paper has no bias (Table 9). MORPH adds one with no weight decay, which could let a
static [0, 0, 3.9, 0] row be learned for free. Measured on the LXTUL 5k seed 1 checkpoint
(`/home/wolfe/morph-to/checkpoints/morph/slot-spandec-strict-fan4-all-fp01-lsel-joint-rf-lam1-rank-cnorm/step_5000.pt`):
all 12 core HC biases have max abs ≤ 0.02, static `Hpost_row` = [1, 1, 1, 1] ± 0.01,
static `Hres − I` = 0.00. Only `prelude.0` has a visible static part (Hpost_row
[1.07, 0.95, 1.00, 0.97], Hres−I 0.05). So the bias is not the source here [ran]. I read
one checkpoint only; the snap and hcuni checkpoints were not read.

### F5 (DEVIATION, medium relevance): pre-norm block, no carrier norm, plus a per-stream pass norm

- Paper (Fig. 2) normalises after the merge (post-norm). MORPH normalises only the
  sublayer's input `x_bar` (row 10), so the carrier norm is unconstrained inside a pass; the
  filing reports the core attention writing an RMS-51 shared output, which a post-norm
  block would have rescaled every sublayer. Whether the paper's post-norm acts per stream or
  over the flattened N is not stated, and either way it would not remove a zero-sum pattern;
  it would bound its size relative to the per-slot part per block, not per pass [inferred].
- LXTUL's per-pass cell RMSNorm (row 14) is per stream over C: each stream ≈ ±v + small, so
  it rescales all four streams by about the same factor and keeps the sign pattern. It
  "divides the blow-up back out" (the filing's words) once per pass, which lets the
  constant regrow every pass without detonating. A single-stream Parcae core has no D at
  all, so none of F1-F5 can happen there; that is consistent with its 2.7x larger loop
  contribution but does not prove it is the cause [inferred].

### F6 (DEVIATION, low relevance in training): exact Cayley lifts the paper's implicit near-identity cap; bf16 Hres is biased expansive

- The paper's s = 2 iteration is only orthogonal while ‖(α/2)W‖ < 1. Measured
  (fp64): orthogonality error 5.9e-6 at raw entry std 0.1, 4.3e-2 at std 1, and it explodes
  at std 5 (σmax 7.7) and std 20 (σmax 690) [ran]. So a trained paper model is forced to
  keep H_res near the identity (rotation angle < 90°). MORPH's exact closed form matches the
  true Cayley to 1e-15 at every scale [ran] and allows any rotation in SO(4), including
  ones that move M fully into D (F2 leak up to 1.0). This deviation is documented (as a
  divergence fix) and it is correct, but it removes an implicit regulariser the paper had.
  The LXTUL checkpoint's static Hres is the identity, and I did not measure the dynamic
  Hres magnitudes on real slot-loop inputs.
- `Hres` and `Hpost_row` are cast to the carrier dtype before the write
  (`hyper_connections.py:213-214`). In training the carrier is fp32 (per the 2026-08-21
  probe note), so this is harmless there. If any path runs a bf16 carrier (decode,
  deploy), rounding a near-identity rotation to bf16 makes it slightly EXPANSIVE: mean
  σmax 1.0009-1.0015, positive mean log-det at small angles, and 96 applications of one
  bf16-rounded Hres (skew std 0.02) give σmax 1.07 mean, 1.20 max [ran]. It is a fixed
  bias, not random noise.

### F7 (documentation drift, no numeric effect)

- `hc_cayley_iters` is dead math but a live dispatch switch (row 8).
- Stale comments: `hyper_connections.py:109` ("Cayley fixed-point steps (s). Default 3"),
  `:33` ("Sinkhorn(0)=uniform→use a diagonal bias"), `:216` ("Sinkhorn / non-cayley: eager
  mapping") on the `use_kernel=False` path; the module docstring of
  `fused_hyper_connection.py:14-18` still describes "Cayley 3-iter" kept in eager, which the
  round-2 fused premap replaced.
- "5.2× cheaper" (row 20) and "exact dynamical isometry" for the whole residual (row 18).
- `docs/references/residual-streams/jpmhc/jpmhc.md:6` still says `residual_mode: hc_cayley`,
  a knob that `docs/references.md` says was removed.
- Missing note: the docstring's root-cause citation (row 7).

## What I did not verify

- The fused kernel against the eager reference on GPU (no GPU allowed). Row 17 is from reading.
- The training carrier dtype on 3f9aae2 (taken from the 2026-08-21 probe note).
- Dynamic H magnitudes on real slot-loop activations. F3's "fixed router" account is
  inferred from the probe's energy shares, the code, and the checkpoint's weight row norms.
  The direct test: log `Hpost_row`, `Hres`, and the per-slot variance of the H logits per
  core layer and pass in `dataflow_probe.py` on the LXTUL 5k checkpoint.
- Whether the paper's post-norm acts per stream or on the flattened carrier (Fig. 2 is
  ambiguous) and what the paper's "s" in eq. (42) is.
- Only one checkpoint was read for the biases (LXTUL 5k seed 1).

## Suggested tests that follow from this audit (not run)

1. Doubly-stochastic or 1-preserving H_res in the slot-loop core only (H_res·1 = 1, e.g. a
   Cayley rotation restricted to the 3-dim complement of 1, or the removed Sinkhorn arm).
   This keeps M and D apart and damps D (F2).
2. Zero-sum-free write: Hpost_row fixed to 1 (uniform write) for BOTH attention and MLP
   HCs in the core (the twin made only the attention HC uniform; the MLP mixers rebuilt the
   constant).
3. Coefficient input from the stream MEAN only (or with D projected out), so a D-vector
   cannot steer the mappings (F3).
4. n = 1 in the slot-loop core (no D at all) on MORPH itself. `model.core_impl: parcae`
   (arm `slot-mnext-parcae-core`) already collapses to the stream mean at core entry, but it
   also swaps attention, MLP and ternary at once; an HC-only n = 1 core would be the
   single-factor twin.
