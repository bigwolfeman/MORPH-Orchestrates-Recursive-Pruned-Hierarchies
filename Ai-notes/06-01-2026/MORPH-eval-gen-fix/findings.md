# MORPH eval + generation re-enable & fix (2026-06-01 ~16:50)

## Trigger
Wolfe killed the dense 15k A/B campaign: *"It wasn't doing eval or testing generation."*
The campaign had `eval_every=999999 gen_every=0` — a sterile throughput/mem benchmark
that could diverge or break AR generation with no signal until the end. The monitoring
mandate (catch divergence, confirm AR generation healthy, #190) requires eval+gen ON.

## Why eval/gen had been disabled — and why that reason is now resolved
The trainer comment (train.py final-eval block) blamed `evaluate()`'s
`model.eval()/.train()` + grad-mode toggle for a "recompilation storm that wedges on
sm_120" (the eval-recompile-hang folder). Re-testing with the CURRENT mitigations:

1. **Eval is actually safe.** `torch.compiler.set_stance("eager_on_recompile")` (set after
   warmup) makes any guard miss from the eval()/train()/grad-mode flip run *eager* instead
   of recompiling. Smoke tests (`smoke_eval_gen2`, `smoke3`, compile on, use_kernels=true):
   `[VAL 20]`, `[VAL 40]`, `Final val` all print, run exits clean. No storm.

2. **Generation hang ROOT-CAUSED (and it is NOT the rare startup timing race).**
   - Smoke test wedged at the step-40 generation, 90s faulthandler caught the main thread in
     `embeddings.py:55 _log_map_origin`. **Red herring:** that fn is straight-line elementwise
     (`acosh/sqrt/clamp/where/*`) — no loop, no `.item()`, no data-dependent control flow, so
     it cannot hang; the snapshot just froze a GPU-bound frame.
   - **Isolated probe** (`ignore/gen_isolated.py`, step_7500 ckpt, pure eager, no compile):
     generation is **~42 ms/tok, perfectly stable** as seqlen grows, finishes ~10s, coherent
     English, **no EOS-loop/repetition degeneration** → #190 partial PASS.
   - **Therefore the wedge was the `torch.compile` interaction:** generation runs token-by-token
     at B=1 with seqlen growing by 1 each step → a brand-new shape every token → under
     `eager_on_recompile` dynamo still pays per-token guard-eval to route each novel shape to
     its eager fallback. >10× slower; over ~300 forwards it blew past the 90s watchdog.

## The fixes (all in `morph/training/train.py`)
1. **`@torch.compiler.set_stance("force_eager")` on `run_generation_test`** — forces the whole
   gen call tree (incl. the training-shape-compiled MLPs) onto the original eager code
   (~42 ms/tok), auto-restores the prior stance on return. Eval (fixed full-batch shape) keeps
   the compiled path. **Verified end-to-end:** smoke_eval_gen2/smoke3 ran VAL+periodic GEN+
   final GEN to clean exit, no wedge.
2. **Throughput metric correctness** — `t_start` reset moved to the END of the loop body
   (after logging/eval/gen/ckpt) so those non-train blocks no longer inflate the NEXT step's
   `_dt`. `steps_per_sec` is now pure train-step compute regardless of eval cadence.
   (`peak_mem` was already a `max()`, immune.)
3. **Gen samples → sidecar** `checkpoints/morph/<run>/generation_samples.txt`, stdout gets only
   a safe 1-line summary. Generated text is uncontrolled tokens that can contain
   `RuntimeError:`/`Killed`/`Traceback...` and would false-trigger `ab_watch.sh`'s ERR_RE.

## Relaunched (monitored) campaign
`ignore/run_ab.sh` COMMON: `steps=15000 eval_every=500 gen_every=2500 gen_test=true
ckpt_every=7500 lr=min_lr=1e-4`, prune/compact OFF. Arms: dense15k_eager_b4 → kernels_b4 →
kernels_b8. Still `MORPH_DEBUG_STEP=1` (startup timing-race suppressor) + retry backstop +
`ab_watch.sh`. eager(B4) GPU ~29.8G/32G live — ~1.5G headroom, watching OOM.

## Still OPEN (unchanged)
The rare (~5-14%) step-<20 **startup timing race** is a separate phenomenon (autograd thread
hot, 0 compiles) — masked by the suppressor + retry, root cause unproven. NOT pursued per
Wolfe's earlier "get the results" call.

## Verification artifacts
- `ignore/gen_isolated.py` + `ignore/gen_isolated.log` (42 ms/tok, coherent step_7500 text)
- `ignore/smoke_eval_gen2.log`, `ignore/smoke3.log` (eval+gen+clean exit, watcher-safe stdout)
- `checkpoints/morph/smoke3/generation_samples.txt` (sidecar proof)
