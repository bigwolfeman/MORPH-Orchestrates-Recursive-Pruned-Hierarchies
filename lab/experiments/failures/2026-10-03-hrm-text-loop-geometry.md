# Planned: does HRM-Text's two-level loop rotate at fixed scale like Huginn's?

Status: failure

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

**Method amendment, 2026-10-03 02:03 CDT, before any GPU run.** Read
`transformers/models/hrm_text/{modeling,configuration}_hrm_text.py` (transformers 5.15.1
locally, 5.13.0 on the 3070; both ship native `hrm_text` support, no `trust_remote_code`
needed — `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)` loads it
directly, matching the model card's own usage snippet). Checkpoint
`sapientinc/HRM-Text-1B`, revision `22097cbcecdd1301afe30a19a3ee61b96a9863e5`: H_cycles=2,
L_cycles=3, num_layers_per_stack=16 (inflated `num_hidden_layers`=128), hidden_size=1536,
vocab_size=65536, `tie_word_embeddings=False` (lm_head is its own Linear, unlike Huginn).

Architecture facts that make part of the prereg's framing architectural, not learned:

- `z_H` starts at `embed_tokens(input_ids) * embedding_scale`; `z_L` starts at
  `z_L_init`, a single `(hidden_size,)` parameter with `requires_grad=False` — broadcast to
  every token, so unlike Huginn's sampled initial latent, HRM's initial state is
  **deterministic** (no RNG, no `fork_rng` needed in the probe). The checkpoint's
  `z_L_init` is NOT the zero `_init_weights` starts a fresh model at: norm 38.75 (close to
  but not exactly `sqrt(1536)=39.1918`), so it moved during training before training froze
  it — an empirical fact, not architectural.
- Every `L_module`/`H_module` call (`HrmTextStack.forward`) ends in `final_norm`, a
  **parameterless** RMSNorm (`x * rsqrt(mean(x^2)+eps)`, no learned gain — the model card
  calls this "Parameterless Pre-RMSNorm"). So every captured `z_H`/`z_L` leaving its stack
  has per-token RMS exactly 1 (up to eps/bf16), L2 norm `sqrt(1536)=39.19`, BY
  CONSTRUCTION — the same role Huginn's `norm_4` placement played for G-1/G-4. This makes
  **R-2 architectural** for both states: it will hold regardless of what the loop learned,
  and is not evidence about learned behaviour. (`embedding_scale`=39.191835884530846
  equals `sqrt(1536)` to the digits shown, so the very first `z_H` entering the loop
  already sits at the same norm scale the post-norm path enforces.)
- `HrmTextModel.forward` returns the H_module's last output directly as
  `last_hidden_state` with no further norm; `HrmTextForCausalLM.forward` applies `lm_head`
  straight to it. So the Huginn-probe's `predict_from_latents` chain
  (`ln_f -> coda -> ln_f -> lm_head`) collapses here to one call: `logits =
  lm_head(z_H_state)`. Faithfulness is therefore a direct check of that one line.
- `self.config.H_cycles` is read inside the Python `for` loop that drives the recurrence,
  and `L_bp_cycles` grad-routing (the only other place `H_cycles` matters) is a
  `torch.no_grad()` choice that is a complete no-op under inference. Monkey-patching
  `model.config.H_cycles` upward (verified `model.config is model.model.config`, so one
  patch reaches the loop) and calling `forward` again is architecturally safe for reading
  CE at depths beyond the trained 2 — PROVIDED `use_cache=False` on every call: with no KV
  cache, `past_key_values` stays `None`, the only read of `cycle_offset` (gated on
  `past_key_values is not None`) never fires, and every stack call recomputes full
  self-attention over the current hidden states, so there is no cache-slot sizing hazard
  from the extended loop (the `DynamicCache` was pre-sized for the ORIGINAL H_cycles=2 at
  construction and would be too small for H_cycles=4 if caching were enabled). This lets
  one extended forward (H_cycles set to 4, 2x trained) give R-1's whole 1x/2x sweep: the
  trajectory up to h=2 is identical to the natural, unmodified run (no lookahead
  dependency), verified below.
- Capture point: `L_module`/`H_module` are genuine `nn.Module` submodules with real
  `forward()` methods invoked through ordinary `__call__`, so the probe uses
  `register_forward_hook` on `model.model.L_module` / `model.model.H_module` (the
  official, additive, non-mutating mechanism) rather than the `__dict__`-shadow trick the
  Huginn probe needed for its plain-method `core_block_forward`.

PrefixLM / plain-text decision: the model card documents `token_type_ids==1` marking a
bidirectional "prefix" block (falling back to pure causal, "noticeably worse", if
omitted) and four single-token condition tags, comma-composable: `direct` (NLP tasks,
few-shot), `cot` (chain-of-thought), `synth` (synthetic/curated), `noisy` ("noisy /
web-crawl style"). There is no documented example for plain continuation with no
condition at all; `noisy` is the one documented tag that matches raw web-crawl text, which
is exactly what OpenWebText is. Decision: prepend `<|im_start|>` (BOS, id 6) +
`<|quad_start|>` (the `noisy` tag, id 12) as a 2-token prefix, `token_type_ids=1` on both
(the card's bidirectional-prefix envelope), `token_type_ids=0` (causal) on the body. Labels
that would score predicting a prefix token are set to -100 (excluded), so 1023 of the 1024
row positions are scored per row (one fewer than Huginn's 1024, recorded in the output) —
every real-text next-token prediction is still scored, including the first body token
given only the 2-token prefix.

Row construction: the SAME 64 Huginn rows (`lab/huginn/huginn_loop_geometry.py::load_rows`,
same dataset glob, seq=1024, skip_samples=0) are decoded back to text with Huginn's own
tokenizer, then re-tokenized with HRM's own tokenizer (`add_special_tokens=False`) to get
a token count per row (HRM's 65536-vocab BPE will not match Huginn's token count
1-for-1); the first 1023 HRM body tokens are kept (1022 as model input, 1 as the last
label) after the 2-token prefix, for a fixed row length of 1024, matching Huginn's. A row
whose HRM tokenization is shorter than 1023 tokens raises, loudly — not silently padded.

Host for the GPU run (changed 2026-10-03, Wolfe): the RTX 3070 (`wolfe@3070`, sm_86, 8 GB,
bf16, torch 2.12.1+cu130, transformers 5.13.0), NOT the local 5090 — the 5090 stays free
for the training queue and this probe does not take `gpu.lock`. HRM-Text-1B is ~2.3 GB in
bf16 and fits without quantisation (quantising would change the very states being
measured). The 3070 carries exactly one OWT shard (`openwebtext-train-00000-of-00080.arrow`,
byte-identical to the Spark's copy) by design, so the 64-row text is independently
reconstructed there from that single shard and SHA-256-hashed against the same hash
computed locally (multi-shard cache); a mismatch stops the run. Faithfulness is checked on
the 3070 itself, not only locally. Both transformers versions (5.15.1 local / 5.13.0
remote) are recorded in the output.

## Results

Run 2026-10-03 on the 3070 (bf16, transformers 5.13.0), 64 rows, 121 s. Row text SHA-256
`05661f36...6932` on both hosts. Faithfulness: max abs logit difference 0. Artifacts:
[`../results/2026-10-03-hrm-text-loop-geometry/`](../results/2026-10-03-hrm-text-loop-geometry/).
Trained depth is H_cycles 2 x (L_cycles 3 + 1); the probe extends to 4 H cycles. CE is per
HRM token and is not comparable with Huginn's CE (different tokenizer).

z_H per H cycle:

| H cycle | norm | cos to v_H | step / norm | step share along v_H | PR | CE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 39.19 | 0.472 | | | 55.5 | 6.516 |
| 2 (trained) | 39.19 | 0.232 | 1.141 | 0.065 | 66.5 | 3.098 |
| 3 | 39.19 | 0.207 | 0.544 | 0.010 | 72.6 | 3.532 |
| 4 | 39.19 | 0.203 | 0.647 | 0.012 | 79.2 | 4.829 |

z_L over its 12 steps: norm fixed at 39.19; cos to v_L 0.30 to 0.69; step share along v_L
0.03 to 0.25; PR 36 -> 229 at step 5, then down to 125 at step 12.

Interventions at the trained depth (CE delta, 95 % paired CI):

| state | rescale | remove v | remove a random direction |
| --- | --- | --- | --- |
| final z_H | +0.00001 [-0.00003, +0.00005] | +0.406 [+0.386, +0.427] | +0.00001 [-0.00023, +0.00024] |
| final z_L (through one H_module call) | +0.00002 [-0.00015, +0.00019] | +0.035 [+0.033, +0.038] | +0.00001 [-0.00020, +0.00021] |

| prediction | reading | held |
| --- | --- | --- |
| R-1 CE at trained depth below 1 H cycle by > 0.1 | 3.42 nats lower | yes |
| R-2 norm stable | ratio 1.000 | yes, architectural (parameterless RMSNorm ends every module call) |
| R-3 H step share along v_H < 30 % from cycle 2 | max 0.065 | yes |
| R-4 H relative step smaller than L's | H 1.14 vs L 0.63 | no |
| R-5 H PR grows > 30 % | +19.8 % | no |
| R-6 random removal < 0.01 and v_H removal > 0.05 | +0.00001 / +0.406 | yes |

## Verdict

Failure: R-4 and R-5 missed. The hypothesis held in part. Like Huginn, HRM-Text moves its
state at a fixed per-token scale (by architecture), its H steps are almost all
perpendicular to the shared direction (6.5 % at the trained cycle, 1 % after), and its
shared direction is a load-bearing bias (removing it costs 0.41 nats, a random direction
costs 0). Unlike Huginn, it is not a contractive loop: its steps do not shrink (0.54, then
0.65), and CE gets WORSE past its trained depth (3.10 at 2 cycles, 3.53 at 3, 4.83 at 4).
HRM-Text is a fixed-depth unrolled network, not a fixed-point iteration. Its "slow" H
state takes the larger steps (it runs only twice), and its fast L state's diversity rises
to step 5 and falls after.

## Updated hypothesis

Across three looped models, the parts that hold are architectural or near-architectural:
a fixed per-token scale on the carried state, steps perpendicular to a small shared bias
direction, and a bias whose removal is costly. Both earning loops have a per-step norm on
the state; MORPH's slot cells have none and grow along their bias. That supports the
per-pass cell norm arm. Contraction and depth extrapolation are NOT universal: Huginn
(trained with a random depth draw, mean 32) has them, HRM-Text (fixed depth 2) does not.
That matches the 2026-09-14 finding that sampled depth and fixed depth learn different
things.
