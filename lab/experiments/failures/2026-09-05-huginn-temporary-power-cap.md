# Planned: run-scoped Huginn power cap

Status: failure

## Question

Can the launcher limit this run to 92 percent of default GPU power and restore
the observed previous cap without depending on the ML process surviving?

## Hypothesis

A system service can apply the cap before starting the unprivileged evaluator.
Its ExecStopPost can restore the previous cap after normal exit, start failure
or process termination while systemd and the NVIDIA driver remain operational.

## Predictions

Frozen before checks on 2026-09-05:

- A 575 W default produces a 529 W cap. A lower existing cap is never raised.
- Unit construction gives administrator privileges only to the NVIDIA setters.
- A cap readback mismatch prevents the evaluator from starting.
- Resume preserves old launch logs and completed-depth checkpoints.
- If administrator authorization is available, the real service applies 529 W
  and restores the observed previous cap after a short non-ML lifecycle check.

## Method

Run focused CPU tests for calculations, command construction, startup refusal
and resume preservation. Use the same service construction with a short sleep
command to check actual hardware application and restoration if authorization
is available. Then resume Huginn with the capped launcher. Keep H1-H6 and the
evaluator source unchanged. This is an operational amendment, not a new quality
experiment. Do not start CUDA work if the cap check fails.

## Limits

Power loss or driver failure may prevent restoration. The cap is GPU board
power, not whole-machine wall power. A 529 W cap does not guarantee UPS safety.

## Method deviation before hardware check

The CPU predictions were written before the tests, but the file was not committed
before that command. This violates the preregistration commit rule. The command
reported 10 passed, including the four existing depth-sweep tests. Those results
are implementation checks, not a compliant preregistered result. The hardware
prediction remains untested and this record is committed before its execution.

## Results

The CPU command `python -m pytest tests/test_huginn_power_limit.py
tests/test_huginn_depth_sweep.py -q -p no:cacheprovider` reports ten passing tests.
It checks cap calculations, the unprivileged service command, cap-mismatch startup
refusal and preservation of real temporary resume files. It does not exercise
the privileged hardware path.

The attempted hardware lifecycle test failed before starting the service.
`pkexec` reported `No authentication agent found` and exited 127. The outer check
exited 1. The GPU cap remained 575 W. Huginn remained stopped. No hardware cap or
restoration result is claimed. The launcher now defaults to terminal sudo
authorization. The invoking user must enter the password in their own terminal.

## Verdict

Failure to complete the hardware validation because administrator authentication
was unavailable. The CPU tests also have the preregistration commit deviation
recorded above. The code is implemented, but real cap application and restoration
remain unverified. H1-H6 and the evaluator source remain unchanged.

## Updated hypothesis

Installed systemd documentation supports ExecStopPost after start failure and
SIGKILL. The constructed unit uses that behavior and limits root privileges to
the NVIDIA setters. Actual behavior on this host still requires an authenticated
hardware lifecycle check.
