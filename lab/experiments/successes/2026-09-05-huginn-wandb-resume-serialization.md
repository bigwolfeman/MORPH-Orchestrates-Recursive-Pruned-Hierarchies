# Planned: Huginn W&B resume serialization repair

Status: success

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

## Results

The two Huginn test files report 12 passing tests, exit 0. The installed W&B Config
class reproduces the integer-key failure. Canonical JSON fixes the equivalent
configuration, while a changed model value still raises ConfigError.

The migration ran with exit 0 against the interrupted checkpoint. An independent
comparison checked the backup SHA256 and exact payload equality after removing
only the audit metadata and restoring the old evaluator source hash. All other
fields are unchanged, including the ten completed depths through K32, CE sums,
profiles, row hashes, configuration fingerprint and W&B ID. The backup path is
stored in the checkpoint's `resume_migrations` field. Reapplying migration and
unrelated source changes are rejected by contract tests.

The evaluator logs the migration audit to W&B on its next resume. It does not use
blanket `allow_val_change=True` for model configuration updates. Only the exact
logging repair is permitted by the migration helper; ordinary source checks remain.

## Verdict

Success for the serialization and result-preservation checks. The corrected CUDA
resume still needs the user's administrator authorization to restart the capped
system service. No claim of remaining-depth completion follows from this result.

## Updated hypothesis

The prior startup failure came from W&B serialization, not the power cap or model
evaluation. The real failed attempt's system journal confirms 529 W application
and restoration to 575 W after evaluator exit. SIGKILL cleanup remains untested.
