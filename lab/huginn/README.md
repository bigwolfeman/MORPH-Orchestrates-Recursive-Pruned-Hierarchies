# Huginn launcher

To run with a temporary 92 percent GPU power limit:

```sh
/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python lab/huginn/launch.py \
  --power-percent 92 \
  --output ignore/huginn/2026-09-05-corrected resume=true
```

Invoke this as the normal user. A desktop administrator prompt authorizes the
system service. Do not run the Python launcher with sudo. The service runs the
model as the invoking user. Only the NVIDIA power setters run as administrator.

For this 5090, 92 percent of the 575 W default is 529 W. The launcher records the
previous cap and never raises a lower existing cap. It checks the effective cap
before starting the evaluator. `CUDA_VISIBLE_DEVICES` selects the capped GPU.

Systemd runs the restore command after completion, failed startup or process
termination, including SIGKILL. The service owns the evaluator's process tree.
The restore command also runs when that service is stopped. This does not protect
against a failed driver or a machine power loss. The cap controls GPU board power,
not whole-machine wall power. Do not change power settings from another tool while
the permission prompt or capped run is active.

Inspect the system service, not the old user service:

```sh
systemctl status morph-huginn-capped.service
journalctl -u morph-huginn-capped.service
nvidia-smi --query-gpu=power.limit --format=csv,noheader
```

Failed services remain available for inspection. Before relaunching a failed capped
service, clear its failed state with `sudo systemctl reset-failed morph-huginn-capped`.
Check the current power limit first. A failed restore is an error, not a successful
run. The launcher does not install a boot-time cap or a passwordless sudo rule.

Resume preserves the previous `run.log`, `process.json` and cap record in
`launch_history/`. It keeps the completed-depth checkpoint in place. A partially
completed depth restarts from its first row. The same online W&B run continues.
