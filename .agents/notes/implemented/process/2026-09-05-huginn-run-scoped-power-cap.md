# Agent Note: run-scoped Huginn power cap

Status: implemented

## Problem

The Huginn run exceeded the workstation's UPS headroom. Wolfe requests a 92 percent
GPU cap and does not want to remember a later manual reset.

## Decision

The Huginn launcher can create a system service with a temporary GPU power cap.
It records the observed previous cap. ExecStartPre applies the requested limit;
ExecStopPost restores the previous limit. The model runs as the invoking user.
Only the NVIDIA setter commands use administrator privileges. A normal terminal
sudo prompt authorizes service creation. No passwordless rule or boot-time cap is
installed. The child checks the applied cap before starting CUDA work.

The cap is a percentage of default board power. The 575 W default gives 529 W at
92 percent. A lower existing cap is preserved. The service owns the complete
evaluator process tree. Its stop command can therefore restore after a killed
supervisor as well as normal completion while the driver and system manager work.

Resume archives previous launch logs without replacing completed-depth results.
The evaluator source and frozen Huginn predictions remain unchanged.
Usage is in [the launcher README](../../../../lab/huginn/README.md).

## Alternatives considered

- Manual cap and reset: Wolfe may forget the reset.
- Python finally or a shell trap: SIGKILL bypasses cleanup in the killed process.
- Run the entire model as root: unnecessary privilege for remote model code.
- Desktop pkexec prompt: this agent session has no authentication agent. Terminal
  sudo is the default; pkexec remains an explicit option.

## Consequences

The initial hardware check could not authenticate. A later user-authorized run
applied 529 W and restored 575 W after the evaluator failed during W&B resume.
The system journal records both transitions at 01:34:47 and 01:35:05 on September 5.
Forced SIGKILL restoration remains untested. The initial test record is
[the operational experiment](../../../../lab/experiments/failures/2026-09-05-huginn-temporary-power-cap.md).

The cap limits GPU board power, not total wall power. It does not guarantee UPS
safety. Power loss or driver failure can prevent restoration. Failed services
retain diagnostic state; check the journal and current cap before retrying.
Do not change another power limit while this service or its prompt is active.

This is separate from the paired-depth evaluation decision. That note remains
active because this change does not replace its scientific or data contracts.
