# Running offline probes on the 3070, beside the Spark

Offline probes (`lab/divergence/*`) read a checkpoint and never train, so they do not need
the 5090. Two eval hosts exist. Use both: on 2026-09-18 the Spark held two semantic probes
plus a depth sweep at 95 % utilisation and that sweep printed nothing in 35 minutes, while
the same sweep finished on the 3070 in about 7.

| host | reach | GPU | notes |
| --- | --- | --- | --- |
| DGX Spark | `ssh dgx-spark` | GB10 | checkout `/home/wolfe/morph-instruments/MORPH`, `./.venv/bin/python` |
| 3070 box | `ssh wolfe@3070` | RTX 3070, 8 GB | checkout `/mnt/bigdata/morph-instruments/MORPH`, `~/morph-venv/bin/python` |

Measured on the 3070 for a 270M code-target arm: depth sweep 480 rows x 7 depths in about
7 minutes at 3.3 GB; semantic probe 120 cuts in roughly 45 minutes at 2.6 GB. Two probes run
side by side at about 5.9 GB. A third risks the 8 GB card, so keep it to two.

## Running one probe

    /home/wolfe/morph-scratch/arc/probe3070.sh <local_ckpt> <label=config> <probe.py> [args...]

It pushes the checkpoint to `/mnt/bigdata`, runs the probe detached with the right
environment, waits for `wrote` or `Traceback`, copies results back, and deletes the remote
copy. For several probes in sequence, write a small chain script that blocks while
`pgrep -cf "[l]ab/divergence"` is 2 or more; keep the bracket so the pattern cannot match
the chain's own command line.

## The environment the probes need

    HF_HOME=/mnt/bigdata/hf  HF_HUB_OFFLINE=1  PYTHONPATH=.  WANDB_MODE=offline
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  OMP_NUM_THREADS=4  MKL_NUM_THREADS=4

## Five things that will cost you an hour

1. **`/home` on the 3070 is full** (447G, about 1.4G free). A checkpoint is 1.2-1.4 GB.
   Everything lives on `/mnt/bigdata` (13T, 3.6T free).
2. **`cfg.data.dataset` is a literal `~/.cache/...` glob and ignores `HF_HOME`.** So
   `~/.cache/huggingface/datasets/openwebtext` is a symlink to the copy on `/mnt/bigdata`.
   Without it every probe dies inside `create_dataloader` with "Couldn't find any data file".
3. **Both probe hosts carry exactly ONE OWT shard**, `openwebtext-train-00000-of-00080.arrow`,
   498140816 bytes, byte-identical on each. That is deliberate: both stream the SAME rows, so
   a 3070 number is comparable to a Spark number. `code_semantic_pair.py` enforces this, since
   it matches the true spans in the sibling `.txt` files line for line before it pairs, and it
   has accepted a 3070 probe against a Spark reference. Adding shards to one host silently
   breaks every cross-host comparison. The 5090 has all 80; the probe hosts have one.
4. **The 3070 is sm_86**, so the SM120 Triton kernels fall back and the banner reads
   `kernels=EAGER+TGSCOPED`. Correct there, not a fault.
5. **`~/morph-venv` has no pip** (torch 2.12.1+cu130). `scipy` is absent and no probe in
   `lab/divergence` imports it. The old `~/MORPH` on that box is a July 3 snapshot and not a
   git repo; ignore it.

## Rebuilding the 3070 checkout

    git bundle create /tmp/morph.bundle master          # about 336 MB
    scp /tmp/morph.bundle wolfe@3070:/mnt/bigdata/morph-instruments/
    ssh wolfe@3070 'cd /mnt/bigdata/morph-instruments && rm -rf MORPH && mkdir MORPH && cd MORPH \
        && git init -q && git fetch -q ../morph.bundle master && git checkout -q FETCH_HEAD'

Ship the checkout BEFORE arming anything that reads it. A probe-loader fix once reached the
Spark 11 minutes after a watcher had already launched the probe it was meant to fix, and the
reading had to be thrown away.
