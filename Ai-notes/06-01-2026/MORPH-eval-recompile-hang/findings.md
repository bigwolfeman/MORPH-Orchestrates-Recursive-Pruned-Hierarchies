# ★★★ TRUE ROOT CAUSE — RESOLVED & VERIFIED (2026-06-01 09:45) — SUPERSEDES ALL BELOW ★★★

Everything below this banner was the investigation trail; some intermediate "RESOLVED"
claims were premature/wrong (they blamed torch.compile fork races + active-set dynamic
shapes). The py-spy stack Wolfe captured on the wedged campaign was decisive and redirected
the whole diagnosis. Final, verified account:

**Mechanism.** The fused-CCA attention kernels are hand-written `@triton.jit` autograd
Functions (`fused_csa_attention`, `fused_hca_attention`, `fused_window_attention`, convs),
NOT `torch.compile` — so `torch.compiler.set_stance("eager_on_recompile")` does NOT govern
them. Triton JIT-compiles a kernel per launch signature and **specializes `size==1`
separately from `size>1`**. The active-set loop (#183) shrinks to `n_active==1` ONLY when one
sequence's Poisson depth is far above the rest — rare. The FIRST runtime `n_active==1` JIT-
compiles the size-1 kernel variant → forks `gcc` to build its launcher stub → if a background
thread (wandb asyncio / HF-streaming httpx) is mid-`malloc` holding the glibc arena lock at
that instant, the forked child deadlocks and the parent's lock is never released → the
autograd engine thread wedges on its next `malloc` (py-spy: `_engine_run_backward` → checkpoint
`recompute_fn` → `_core_step` → CCA attention, GPU pinned, NO compiler frame). Rarity matches
all evidence: 1 campaign hit, 0/24 relaunches (incl. exact cold-cache + wandb-online), 0/600
eager fwd+bwd cycles.

**Why earlier hypotheses were wrong (each falsified by an experiment):**
- "torch.compile backward fork-deadlock" → py-spy showed NO inductor/compile frame; pure recompute.
- "degenerate small-batch n_active=1 hangs the kernel" → `repro_recompute_hang.py`: all sizes
  4,3,2,1 fwd+bwd clean <1.6s.
- "eager (use_kernels=false) attention path" → `use_kernels` only gates the CE head; attention
  always uses the fused Triton kernels. Both repro configs clean.
- "GPU kernel hang on rare data" → `stress_kernel_hang.py`: 600 varied fwd+bwd, slowest 1.31s, clean.

**Fix (train.py, by construction — also protects the future pruning run):**
1. Move model build + `torch.compile` + warmup BEFORE `wandb.init()` and the streaming
   dataloader → the entire compile/`gcc`-fork window is single-threaded → every fork is safe.
2. Warmup FORCES every active-set size incl. `n_active==1` (monkeypatch `_sample_depths` to
   patterns [8,8,8,8]→[8,8,8,1]→[8,8,1,1]→[8,1,1,1]) so ALL Triton variants (fwd+bwd, size-1
   and size>1) compile in that thread-free window → none JIT-compiles at runtime.
3. Keep `eager_on_recompile` as a backstop for any leftover torch.compile MLP guard.

**Verification (positive invariant, NOT "didn't hang" — the bug is too rare for that):**
`ignore/verify_fix.sh` — cold triton+inductor caches, 300 real steps, assert ZERO cache files
written after the warmup-done marker. Result: **triton=0, inductor=0 compiled during the loop**,
sps 1.49, loss 11.57→8.71, exit 0. Plus `repro_recompute_hang.py` run-2 delta=0 (size-1 cached).
Note: the eager (no-compile) repro DOES compile `_fused_window_bwd_*` that the production
(compile-on) path never triggers — an artifact of the no-compile path, irrelevant to prod.

**Campaign relaunched** with the fix: `ignore/run_ab.sh` (task bir21qcaq, watcher buvwgtv9h),
arms eager_b4 / kernels_b4 / kernels_b8, 15k steps. Arm1 healthy @ 31.4 GiB.

---

# MORPH eval@1500 recompile hang — diagnosis + tonight's mitigation (2026-06-01 ~01:30)

## Symptom
First A/B attempt (campaign `bl3wmwvjf`, RUN 1/3 = `dense15k_eager_b4`) trained cleanly
0→1400 (loss 11.44→5.24, ppl 93k→189, sps ~1.5) then **froze at the step-1500 boundary
for ~56 min** with:
- main Python thread `R`, ~98% of one core (GIL-bound) — but `State: S (sleeping)`,
  `wchan: futex_wait` at the instant sampled; cputime kept advancing (utime +460/600 ticks/6s).
- GPU PID sm=96%, mem=48% (actively launching kernels).
- log mtime frozen; zero forward progress.

## Root cause (confirmed from the log, not inferred)
`ab_eager_b4.log:92-94`:
```
W torch._dynamo hit config.recompile_limit (8)
  function: '_activation_hook' (morph/model/titans_core/block_sparse.py:542)
  last reason: 4/3: GLOBAL_STATE changed: grad_mode
```
`evaluate()` (train.py:58) is `@torch.no_grad()` + `model.eval()` … `model.train()`. That flips
**grad_mode AND self.training** — both are torch.compile guard dimensions. At step 1500 the
entire compiled surface (core MLPs compiled with `dynamic=True`, fused-kernel autograd fns,
the `_activation_hook` score hook) is forced to recompile under the new global state. On
sm_120 + active-set variable batch + dynamic shapes this degenerates: it hit recompile_limit,
fell back to eager for `_activation_hook`, and then **wedged** somewhere downstream of the
eval transition.

NOTE — honesty boundary: I confirmed the *trigger* (grad_mode recompile at the eval boundary)
from the log. I could NOT capture the exact *final* hang frame: `ptrace_scope=1` + password
sudo blocked py-spy/gdb (trainer isn't my process ancestor, I'm not root, Wolfe asleep). So
"wedged downstream of the recompile fallback" is the strongest claim the evidence supports;
the precise stuck frame (Inductor/Triton autotune-at-eval? a CUDA-side stall under the
eager-fallback active-set path?) is UNVERIFIED.

## Scope — this is campaign-wide, not eager-only
All 3 arms compile core MLPs (`dynamic=True`) and all call the same `evaluate()`. The eager
arm just hit it first. Every arm would hang at every eval (1500, 3000, …) AND at the
unconditional final eval (train.py:440) → the whole night wasted + comparison corrupted.

## Tonight's mitigation (does NOT pretend to fix the bug)
This A/B's deliverable is throughput + peak memory (tok/s, step/s, alloc/reserved) eager vs
kernel — eval PPL is orthogonal and eval actually *perturbs* peak-mem/throughput. So eval-off
is the correct setup for THIS benchmark and sidesteps the hang:
- `ignore/run_ab.sh` COMMON: `eval_every=999999 gen_every=0` (gen_test already false).
- `train.py:439`: guard the previously-UNCONDITIONAL final eval behind
  `if eval_every <= total_steps:` so eval-disabled runs actually EXIT (else each arm wedges
  after its checkpoint and run_ab.sh never advances). py_compile-checked.
- Checkpoints still save (ckpt_every=7500) → AR-gen sampling (#190) runs later in a clean,
  uncompiled process (so it won't hit this).

## Proper fix (DAYTIME, needs Wolfe / root)
The bug is real and must be fixed before any run that needs eval/gen mid-training. Candidates:
1. **Uncompiled eval**: keep a non-compiled reference to the model (or `torch.compiler.disable`
   region) for evaluate()/generation, so the compiled training graph is never invalidated by
   the grad_mode/eval toggle. Most principled.
2. Investigate WHY it fully wedges (vs just recompiling twice and proceeding) — needs py-spy
   with root (`sudo sysctl kernel.yama.ptrace_scope=0` or run py-spy as root). Suspect
   Inductor/Triton autotuning under no_grad on sm_120, or the active-set `.item()` sync path
   under eager-fallback during eval. UNVERIFIED — get the stack first.
3. Confirm `_activation_hook` (block_sparse.py:542) should be `@torch._dynamo.disable`d — it's
   a stats hook, not perf-critical, and is a recompile magnet. Precedent: neural_memory.update
   was dynamo-disabled for a similar reason (see CLAUDE.md torch.compile section).

## SECOND, DISTINCT HANG — Inductor SubprocPool deadlock (attempt 2, ~01:32→01:56)
The eval-disabled relaunch (`blg6vtvgu`) hung AGAIN, but at a different place and for a
different reason — do NOT conflate with the eval/grad_mode bug above.
- Symptom: completed step 0 (printed), then wedged at step 1. Main Python thread pinned at
  ~100% of ONE core (`futex_wait`), GPU 91%, for 24 min. NO dynamo recompile logs this time.
- Evidence: trainer spawned the Inductor parallel compile pool — `compile_worker/__main__
  --workers=32 --kind=fork`; all 32 workers `S` / 0:00:00 CPU = IDLE. Inductor cache
  (`/tmp/claude-1000/torchinductor_wolfe`) newest file = 00:16 (the FIRST run) → the 2nd run
  wrote ZERO compile artifacts in 24 min → not compiling, wedged. No orphan procs.
- Mechanism: the **forked SubprocPool deadlocks** — main thread spin-waits a compile result
  over the worker pipes (fd 59/62) while workers idle, waiting for work never dispatched.
  Cache-hit on step-0 shapes (reused first run's cache) → step-1 NEW active-set shape →
  cache-miss → dispatch to dead pool → hang. Intermittent (timing): that's why attempt-1
  reached 1400 (compiled live, no pool race) and attempt-2 wedged at step 1.
- Could NOT get the exact stack (ptrace_scope=1 + no root). Signature (idle worker pool +
  spinning main + zero compile output) is the documented torch fork-SubprocPool deadlock.

### Fix applied (attempt 3, `bg0x27t8p`, launched 02:02)
- `ignore/run_ab.sh`: `export TORCHINDUCTOR_COMPILE_THREADS=1` → synchronous in-process
  compile, no SubprocPool, no deadlock surface. torch.compile stays ON (throughput stays
  representative); cold-start is slower (single-thread compile).
- Cleared `/tmp/claude-1000/torchinductor_wolfe` (removed partial/locked entries + the
  cache-hit/miss asymmetry that triggered it).
- VERIFICATION PENDING: must confirm attempt-3 trains PAST step 1 (and ideally to step 200+).
  Discriminator if it looks stalled: Inductor cache write-activity (single-thread compile of
  step-0 shapes legitimately takes several min) vs zero writes = wedged.

## Process-management footgun hit while killing (note for future)
`pkill -9 -f "_inductor/compile_worker"` MATCHED THE KILL COMMAND'S OWN SHELL (the pattern is
in its argv) → the script SIGKILLed itself (exit 1, truncated output). Use a non-self-matching
pattern (`compile_worker/__main__` + `grep -v $$`) or pgrep→kill-by-PID excluding own PID.

## attempt-4 (compile=false) — NOT a hang, just unusably slow (verdict corrected)
Disabled torch.compile entirely. Suspected a 4th hang (stuck at step-0 print, one core
pegged) — but the wandb binlog GREW 65536→163840 B over 10 min → it WAS advancing, just
crawling at ~0.2 sps (≈22h/arm, days for 3 arms) AND unrepresentative (standard config uses
compile). So compile-off is viable-but-useless. Lesson: don't infer "wedge" from one-core+
GPU-busy alone — confirm via wandb binlog growth / step-200 print before judging.

## FINAL config (attempt-5, `bogf7fk33`, watcher `bixsj07r7`) — compile ON + SPAWN pool
Re-enabled compile. The four attempts triangulate the fix:
- attempt-1: compile ON, fork pool, FRESH cache → WORKED (1400 steps @ sps 1.4); only died at
  eval@1500 (separate grad_mode bug, now fixed by eval-off).
- attempt-2: compile ON, fork pool, REUSED/partial cache → fork-SubprocPool deadlock @ step 1.
- attempt-3: compile ON, COMPILE_THREADS=1 (single-thread) → wedged in Triton-launcher compile.
- attempt-4: compile OFF → works but 0.2 sps, unusable.
→ Winning config = attempt-1 minus its faults: compile ON + `TORCHINDUCTOR_WORKER_START=spawn`
  (spawn workers carry no inherited CUDA/lock state → no fork deadlock, still parallel/fast) +
  eval-off + FRESH cache each launch. VERIFICATION PENDING: must see step 200+ at sps≈1.4.
  If spawn ALSO deadlocks/hangs → the compile machinery is genuinely broken on this updated
  CachyOS toolchain (real env regression) and the only tonight-option is compile-off-and-accept
  -slow or defer the A/B to Wolfe.

## ✅✅ RESOLVED (2026-06-01 ~08:17) — root-caused with TORCH_LOGS, fixed in-code, verified ✅✅

ROOT CAUSE (definitive, from `TORCH_LOGS=recompiles` on the real trainer): torch.compile
RECOMPILES leak into the training loop and fork-deadlock against background threads. The
recompile guards that leaked past the warmup: `GLOBAL_STATE changed: grad_mode` (active-set
no_grad/grad split), `dtype mismatch fp32↔bf16`, and `size 0/1` — all concentrated on the CMS
block-sparse modules, and the **`_activation_hook` forward-hook was the dominant leak source**
(re-specialized every step). Each leaked recompile forks gcc/Inductor-workers while wandb's
asyncio/status threads (+ HF-streaming httpx, inductor read-thread) hold a glibc malloc lock →
deadlock in `__triton_launcher.c`. Intermittent (fork timing) — which is why attempt-1 ran 1400
steps and others wedged at step 1-30.

UPDATE (~08:35): the 3-part fix below was NOT fully bulletproof — spawn workers do NOT cover
the Triton LAUNCHER (.c) gcc compile, which forks in the MAIN process. A rare Poisson-depth
draw still recompiles intermittently (the hook-disable removed most, but not all), and that
launcher-fork deadlocked the real campaign at step 1 while 160 diag steps ran clean (luck).
FIX COMPLETED with a 4th layer: **`torch.compiler.set_stance("eager_on_recompile")` after the
warmup** — forbids ALL training-loop compilation, so the rare uncovered shape runs EAGER (no
fork, no deadlock — categorically impossible) instead of recompiling. Works now (vs the earlier
failed attempt-9) BECAUSE the warmup+hook-disable cover the common shapes, so the stance only
catches the rare residual → throughput stays full. VALIDATED: 300 steps, wandb ONLINE, exit 0,
sps 1.61 steady, loss 11.54→8.69, no stall. Real campaign relaunched (task b5blskqwl).

THE FIX (4 parts, all in committed code — fixes ANY compiled run incl. pruning):
1. **`@torch.compiler.disable` on `_activation_hook`** (block_sparse.py:542) — it's a non-perf
   stats hook and was the main recompile magnet. Removing it from the graph → **0 recompiles
   after warmup** (measured: was 9/30 steps → now 0/100 steps).
2. **Inductor `worker_start_method="spawn"`** (train.py compile-safety block) — spawn workers are
   fresh processes that never inherit the main thread-lock state → compilation can't fork-deadlock
   even if a residual recompile fires. Belt-and-suspenders.
3. **Fixed 24-pass single-threaded warmup** before the dataloader (train.py) — does the one-time
   bulk compile while the process is single-threaded. (Replaced the flaky adaptive early-exit;
   removed the harmful `eager_on_recompile` which forced permanent eager slowness.)

VERIFIED: real train.py eager B4, 100 steps, exit 0, final checkpoint, **0 post-warmup recompiles**,
~2 sps, no env-var dependency (in-code). Controlled harness ignore/repro_deadlock.py reproduced
the deadlock (mode=none) and confirmed the fix (mode=warmup + spawn). Diagnostic scaffolding:
ignore/repro_recompiles.py, repro_deadlock.py, repro_compile_hang.py, triton_min_test.py
(toolchain proven healthy: 0.32s) — DELETE these + clean run_ab.sh comments when committing.

STATUS: real A/B campaign relaunched 08:17 (run_ab.sh, watcher bwijczoiw). Verifying eager_b4
crosses 200+ then kernels_b4/b8. Memory note for Wolfe's question: eager ~31 GiB is the
full-logits CE (`[16384,49152]`×3) by design; kernel arms use fused-CE → ~12.5 GiB (the A/B
headline). #186 unblocked; #190 (AR-gen) once step_15000 ckpt exists.

---
## ★ FINAL HANDOFF TO WOLFE (2026-06-01 ~03:32) — A/B BLOCKED [SUPERSEDED by RESOLVED above] ★

Bottom line: the A/B campaign did NOT produce numbers. The dense eager-B4 baseline arm never
got past step ~1. After 9 launch attempts + isolation repros + a faulthandler stack, the root
problem is an interaction between the **active-set dynamic shapes + torch.compile + the
streaming HF dataloader** that has TWO faces, and escaping one lands you in the other:

  • Compile DURING the training loop → forks gcc/Inductor workers while the HF streaming
    dataloader's httpx/connection threads hold a (glibc malloc-arena) lock → child deadlocks
    in `__triton_launcher.c`. Intermittent (depends on fork timing). This is what wedged
    attempts 2/3/5/6/8 (cache writes observed DURING the wedge).
  • Forbid recompiles (`torch.compiler.set_stance("eager_on_recompile")`, attempt-9) → no
    fork, no deadlock, BUT any active-set shape not pre-compiled in warmup runs FULLY EAGER →
    step-1 forward crawls for >4.5 min (262M looped model, up to 8 loop iters, eager, S=4096).
    faulthandler stack (the definitive evidence) caught the MAIN thread MOVING through normal
    fast ops across two dumps:
        dump@90s : fused_cca_conv.py:440 causal_conv_reference  (2× F.conv1d — fast op)
        dump@135s: torch/nn/modules/linear.py:134 forward       (F.linear — fast op)
    → NOT a lock deadlock; it's executing+progressing but the step never finishes in
    reasonable time. (Process then hit SIGSEGV 139 — faulthandler+CUDA instability from the
    repeat dumps, NOT the real bug.) No Inductor cache writes during the wedge ⇒ compile ruled
    out for attempt-9; data pipeline ruled out separately (repro fetched 6 streaming batches
    @0.01s each). So step-1 slowness = eager execution of uncovered shapes under the stance.

WHY attempt-1 "worked" (ran 1400 steps @ sps 1.4): it compiled lazily during training and the
fork just happened to miss the lock window that time (luck), and its shapes mostly hit the
compiled cache. The bug is timing-dependent, which is why it took so long to corner.

### What's CONFIRMED FIXED (correct, keep) vs PARTIAL
- eval@1500 grad_mode recompile hang → FIXED: eval-off for this A/B + final-eval guard (train.py:~440).
- Initial-compile fork-deadlock → FIXED by single-threaded multi-pass warmup pre-dataloader (train.py:~271-300).
- Per-step recompile fork-deadlock → WORKED AROUND by `eager_on_recompile` (train.py:~305) — but
  this causes the unusable step-1 slowness above. NET: code no longer DEADLOCKS, but eager-B4
  is too slow to train. Both warmup + stance are still IN train.py (functional, deadlock-free).
- Toolchain is HEALTHY: minimal Triton kernel compiles 0.32s; full model fwd+bwd (no dataloader)
  compiles+runs 10.2s (`ignore/repro_compile_hang.py`). The `_POSIX_C_SOURCE 202405L` warning is
  a RED HERRING (present in successful compiles too). `ignore/repro_data.py` proves data is fast.

### RECOMMENDED REAL FIXES (Wolfe to choose — ranked)
1. **STATIC-shape active-set (best, architectural).** The whole problem is that the active-set
   loop (`transformer.py:_forward_single`, ~line 365-391) feeds the compiled core MLP a DIFFERENT
   sub-batch size + grad/no_grad split each step → endless recompiles. If the active set is PADDED
   to a fixed size (process full B every iter, mask the frozen tail instead of shrinking — i.e.,
   trade the ~25% FLOP active-set saving for shape stability), the core compiles ONCE in warmup,
   never recompiles → fast AND deadlock-free. Remove `eager_on_recompile` then. This is the clean
   fix; it just gives back the active-set speedup (A1, task #183).
2. **Non-streaming / preloaded dataloader (surgical, less code).** The deadlock is fork-vs-httpx
   -threads. If `data.py` doesn't stream (pre-tokenize a shard to a local .bin/memmap, or
   `load_dataset(streaming=False)` for a subset), there are no background httpx threads → compile
   CAN fork safely during training → remove `eager_on_recompile`, keep dynamic shapes, get fast
   compiled training. Cost: download/pre-tokenize step.
3. **Just answer "do the KERNEL arms even run?"** The eager baseline (use_kernels=false) is the
   slow one (eager attention refs). The kernel arms (use_kernels=true, the REAL config) were never
   reached (eager runs first in run_ab.sh and blocks). Quick test: run dense15k_kernels_b4 ALONE
   first. If it trains fast, you at least get the kernel-side numbers tonight and can drop/shorten
   the eager baseline.

### ALL 9 ATTEMPTS (chronological, for the record)
 1 fork pool + fresh cache → ran 1400 steps, died at eval@1500 (grad_mode recompile).
 2 + eval-off, fork pool, REUSED cache → fork-SubprocPool deadlock @ step 1.
 3 + COMPILE_THREADS=1 → wedged in Triton launcher compile @ step 0.
 4 compile OFF → trains but 0.2 sps (≈22h/arm), unusable + kernel arms need Triton anyway.
 5 spawn pool → same launcher-compile wedge (independent of pool strategy).
 6 fork + per-run fresh cache → same wedge (contradicted #1 → led to isolation repro).
 7 + single-pass warmup → warmup compiled clean (9.8s!) but step-1 recompile leaked → deadlock.
 8 + 12-pass warmup → cache stabilized but rarer shape still leaked @ step-1 → deadlock.
 9 + warmup + eager_on_recompile → NO deadlock, but step-1 forward crawls (full eager) >4.5min.
 (diag) faulthandler in train.py → captured the step-1 stack (forward executing, not locked).

### Files touched (all in train.py + run_ab.sh + ignore/)
- train.py: warmup-compile block (~271-300), `set_stance("eager_on_recompile")` (~305),
  final-eval guard (~440), `cache_size_limit=64`. (faulthandler diag line REMOVED.)
- ignore/run_ab.sh: default fork pool + per-run cache clear + eval-off + flat LR. NOT committed.
- ignore/repro_compile_hang.py (model, no dataloader → 10.2s OK), repro_data.py (data OK),
  triton_min_test.py (0.32s OK), ab_watch.sh (watcher). All in gitignored ignore/.
- Hung-attempt logs archived: ignore/ab_hung_attempt{1,2,3,5_spawn,6_fork}/, ab_attempt{7,8}_partial/, ab_slow_eager_attempt4/.
- GPU is FREE; no trainer running. wandb 'morph' has aborted runs only (no usable curves).
- NOT touched: eval-recompile root fix (still TODO), the active-set code, data.py.

### Tasks
- #186 (A/B) still in_progress — BLOCKED on the above decision.
- #190 (AR-gen samples) — blocked: no trained checkpoint exists (no run reached step 7500).

## Status (earlier)
- attempt-1 campaign `bl3wmwvjf` killed (eval@1500 recompile hang). Logs → `ignore/ab_hung_attempt1/`.
- attempt-2 campaign `blg6vtvgu` killed (SubprocPool deadlock @ step 1). Logs → `ignore/ab_hung_attempt2/`.
- attempt-3 RUNNING: campaign `bg0x27t8p`, watcher `btpg4in88` — eval-disabled + COMPILE_THREADS=1 + clean cache.
- TWO real bugs to fix properly in daytime: (1) eval/grad_mode recompile (uncompiled-eval /
  dynamo-disable `_activation_hook`); (2) decide whether COMPILE_THREADS=1 is permanent or
  whether to set `TORCHINDUCTOR_WORKER_START=spawn` (spawn pool avoids the fork deadlock while
  keeping parallel compile). Both block the pruning run (needs eval + many recompiles).
