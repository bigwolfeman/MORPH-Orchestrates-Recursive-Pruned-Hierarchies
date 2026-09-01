# Does trace inversion do anything on OpenWebText? Tested 2026-08-18.

Short answer: the model works, it responds to its input, and on prose it produces
**document summaries and prompt meta-commentary, not reasoning**. Its output is not
usable as a reasoning-distillation target on web text.

## First: the published checkpoint is broken as distributed

`Jackrong/Trace-Inverter-4B` presents as a plain model — safetensors, `config.json`, no
`adapter_config.json`, `library_name` unset. Inside, every projection is stored PEFT-style:
**252 `base_layer.weight`, 252 `lora_A`, 252 `lora_B`, and zero merged `*_proj.weight`**
across 902 tensors. `AutoModelForCausalLM.from_pretrained` therefore reports all 252 as
MISSING, newly initialises them, and returns a randomly-weighted network.

Our first run did exactly that and emitted token salad. That was **our loading bug, not
the model** — the same failure class as the QAT parametrize renaming earlier today, where
a state dict whose keys do not match the architecture loads "successfully" and produces
noise. Anyone evaluating this checkpoint the obvious way measures nothing.

Merged by hand: `W = base_layer + (alpha/r) * B @ A`, `r = 64` from the tensor shapes.
`alpha` is unrecoverable (no adapter config was uploaded), so it was chosen empirically
against the reference traces this very model produced, on 3 rows of
`Jackrong/Claude-opus-4.7-TraceInversion-5000x`:

| scaling | mean word-similarity to reference |
|---|---|
| 1.0 (alpha 64) | 0.346 |
| **2.0 (alpha 128)** | **0.421** |

2.0 also reproduces the reference's exact `**Identify the Question:**` formatting and the
`</tool_call>` defect the model card documents. `alpha = 2r` is the LLaMA-Factory
convention. Merge code: `ignore/tul_logs/merge_inverter.py`.

## Control — it works in distribution

Fed its own documented format (problem + final answer + reasoning bubbles), the merged
model reproduces its own published traces at **0.480** mean word similarity under greedy
decoding. That is the bar that had to be cleared before anything on OWT was interpretable.

## OWT — the mapping, and what came out

    problem      = preceding context (180 words)
    final answer = the next span (45 words), i.e. what a slot must predict
    bubbles      = none; prose has none

**The mechanical falsifier passes.** Swapping in a continuation from a *different*
document changes the trace: similarity 0.153 over 5 samples. So the output is conditioned
on the target, not boilerplate.

**But reading the traces, they are not reasoning.** Two representative samples:

* A news passage about a field hospital evacuation produced *"1. Identify the Core Event
  … 2. Identify the Key Participants … 3. Extract the Timeline and Sequence of Events"*.
  That is reading comprehension of the context, not inference toward the continuation.
* A passage about a primary election produced *"1. Identify the Core Task: The user
  provided a prompt asking to 'Reconstruct the full reasoning trace' … Analyze the Prompt:
  Input: A news article snippet … Output: The final answer provided by the model is a
  continuation of the news article."*

The second is the informative one. **The model noticed there was nothing to infer and
narrated the prompt instead.** It describes the target as "a continuation of the news
article", which is an accurate observation and an admission that no reasoning connects
context to continuation.

## Limitation of our own falsifier

The swap test is weaker than it looked. The target span is quoted inside the trace, so a
*summary* of "context + span A" also differs from "context + span B". A low similarity
therefore separates "responds to input" from "fixed boilerplate", but **cannot separate
reasoning about the target from restating it**. The manual read is what settled it, and
5 samples is a small read.

## What this means

Web text mostly has no inferential step between one span and the next, so an inverter has
nothing to invert and falls back to summarising. Traces harvested this way would teach a
student to summarise passages and narrate prompts.

This does not touch arm D, which never depended on reasoning traces — its teacher is our
own model with the tokens still present, and its target is context, not thought. It does
close off the version of the reasoning phase that would have run an inverter over
pretraining data. If reasoning is wanted in the latent, it has to come from data that
contains reasoning.
