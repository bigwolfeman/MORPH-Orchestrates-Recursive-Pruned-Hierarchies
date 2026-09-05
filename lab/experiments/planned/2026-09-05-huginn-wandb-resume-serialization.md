# Planned: Huginn W&B resume serialization repair

Status: planned

## Question

Can resumed W&B logging accept the same effective model configuration after its
dictionary keys have passed through JSON, while still rejecting actual changes?

## Hypothesis

Canonical JSON representation before logging removes integer-versus-string key
differences. It does not require allowing arbitrary configuration changes.

## Predictions

Frozen before tests:

- W&B rejects the original integer-key dictionary against its saved string-key form.
- The corrected logging conversion accepts those equivalent dictionaries.
- A real model-value change still raises a W&B configuration error.
- An audited checkpoint migration preserves every measured result and row identity.
- Migration rejects source edits beyond the exact logging-only repair.

## Method

Use the installed W&B Config class for the regression checks. Keep evaluation
math and H1-H6 unchanged. The migration accepts only the exact old evaluator
blob and the exact new logging-only patch, verifies every other source hash,
backs up the original checkpoint bytes, and records old/new hashes in an audit
field. Apply it to the stopped run only after these checks. Restart requires
administrator authorization for the existing capped system service.

## Not verified before execution

The corrected live CUDA resume and successful completion of the remaining depths.
