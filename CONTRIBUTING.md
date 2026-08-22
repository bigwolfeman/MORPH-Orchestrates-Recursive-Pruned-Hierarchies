# Contributing To MORPH

MORPH is open source research infrastructure for a fast-moving language-model architecture. Contributions are welcome, but the project is not governed like a general-purpose framework where every useful option becomes a permanent flag.

The goal is to keep MORPH coherent: each numbered release should describe one architecture that can be trained, studied, reproduced, and improved.

## Development Model

MORPH uses a stable snapshot plus research integration model:

- `main` is the current stable line. It should stay usable, reproducible, and documented.
- `next` is the active research integration branch. Architecture changes land here first.
- `experiments/<topic>` branches are for focused implementation or ablation work.
- Numbered releases such as `MORPH-1`, `MORPH-2`, and later versions are frozen snapshots cut from a coherent `next` state.

Frozen releases may also be mirrored into separate read-only repositories when that helps citation, reproducibility, or long-term use. Active development should remain centralized unless there is a strong reason to split it.

## What Belongs Where

Bug fixes may target `main` when they preserve the current release behavior. If a bug fix changes training dynamics, checkpoint compatibility, kernels, or public configs, target `next` unless the maintainer explicitly asks otherwise.

Documentation fixes may target `main` when they correct the stable release. Documentation for in-progress research belongs on `next`.

Research changes target `next`. Start with an issue or RFC before opening a large implementation PR.

Architecture changes target `next` and require evidence. A new mechanism should replace an existing mechanism, improve a measured weakness, or justify the permanent complexity it adds.

Performance and kernel changes require both correctness validation and benchmark evidence. Speed without a correctness story is not enough.

## Contribution Lanes

Good first contributions:

- Fix incorrect docs, comments, diagrams, references, or configs.
- Add focused tests for existing behavior.
- Reproduce an existing ablation or benchmark.
- Fix isolated bugs without changing the architecture.

Advanced contributions:

- Training stability improvements.
- Kernel correctness or performance work.
- Quantization and deploy-path improvements.
- Sparse execution, routing, or memory-system changes.
- Architecture proposals backed by ablations or clear experimental plans.

Not accepted without prior discussion:

- Large rewrites.
- New runtime feature-flag branches in hot paths.
- New optional mechanisms that duplicate existing ones.
- Silent fallbacks that hide broken kernels, missing dependencies, or bad states.
- Mock data paths that can affect development or production runs.
- Changes that make `main` less reproducible.

You are free to fork the project and ablate various methodoligies to prove the worth of an idea. Reach out on discord with details on the changes and results. Most contributions will be manually merged.

## Architecture Policy

MORPH is not a collection of every promising paper mechanism. It is a selected architecture.

Every accepted component should earn its place by improving one or more of:

- training stability
- perplexity or downstream quality
- memory efficiency
- throughput
- deployability
- interpretability of the system
- project coherence

Avoid adding a second implementation when the existing one can be fixed. If a new mechanism replaces an old one, remove the displaced logic instead of leaving both paths active.

Do not add silent fallbacks. Operators should know when a kernel, dataset, checkpoint, quantization path, or training feature fails.

## Evidence Expectations

The required evidence depends on risk:

- Small docs/comment fixes need only review.
- Bug fixes need a test, reproduction, or clear explanation.
- Config changes need a reason and should point to the affected training recipe.
- Kernel changes need correctness comparison against a reference path and must use z3 or lean to prove correctness and bit exact accuracy through direct measurement.
- Architecture changes need ablation evidence or an explicit experimental plan.
- Changes to default training behavior need enough evidence to justify becoming part of the next MORPH release.

When reporting benchmark results, include hardware, config, command, commit, and whether kernels or `torch.compile` were enabled.

## Pull Request Checklist

Before opening a PR:

- Target the correct branch: `main` for stable fixes, `next` for research.
- Keep the change focused.
- Read the surrounding code and references before touching shared paths.
- Update docs, figures, or configs if behavior changed.
- Add or update tests when the change affects behavior.
- Run the most relevant validation command you can.
- Call out anything not tested.

For architecture PRs, also include:

- the problem being solved
- the mechanism being changed
- what old logic is removed or simplified
- expected risks
- ablation or benchmark evidence
- whether checkpoints/configs are affected

## Contribution Licensing

MORPH is licensed under the GNU General Public License v3.0.

Unless you explicitly state otherwise in writing, any contribution intentionally submitted for inclusion in MORPH is submitted under GPL-3.0, matching the project license. Do not submit code, weights, data, figures, or text that you do not have the right to license under GPL-3.0 (or terms that allow relicensing under GPL-3.0).

Vendored third-party code must retain its original license notices.

## Release Policy

A numbered MORPH release is cut only when `next` has settled into a coherent architecture. Releases are not just a bundle of commits.

Each release should include:

- frozen source
- canonical config
- architecture diagrams
- training recipe
- known limitations
- benchmark or ablation notes
- migration notes from the previous release

After release, compatibility fixes may land on `main`, but new research resumes on `next`.

## Governance

MORPH uses maintainer-led architecture governance.

Contributors are encouraged to propose, test, and improve the system. Final decisions about what becomes part of a numbered MORPH release remain with the maintainer so the architecture stays coherent and reproducible.

The project is open source, but not open-ended. The standard is not "can this be useful?" The standard is "does this make MORPH better as a single architecture?"

## AI Policy

The use of AI is allowed and encouraged. Eventually this project will be subject to recursive self improvement, if such a thing proves to be reliable and useful.

As of July 2026 LLMs make too many mistakes to be fully trusted with contribution. The LLM user must ensure coherence of ideas and results and take accountability of every line of contribution.

LLMs should be instructed to not use branching in the hot path unless necessary.
