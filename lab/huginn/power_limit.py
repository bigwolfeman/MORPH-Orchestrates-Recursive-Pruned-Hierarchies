"""Run-scoped GPU power limits, owned by systemd rather than the ML process."""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess


@dataclass(frozen=True)
class PowerPlan:
    uuid: str
    previous_watts: float
    default_watts: float
    limit_watts: int
    percent: float


def query_power(gpu: str) -> tuple[str, float, float, float]:
    result = subprocess.run([
        "nvidia-smi", "-i", gpu,
        "--query-gpu=uuid,power.limit,power.default_limit,power.min_limit",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True)
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("power cap requires exactly one GPU")
    uuid, current, default, minimum = [s.strip() for s in rows[0].split(",")]
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid):
        raise ValueError("invalid GPU UUID")
    values = tuple(float(v) for v in (current, default, minimum))
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("GPU does not report finite positive power limits")
    return uuid, *values


def make_plan(gpu: str, percent: float) -> PowerPlan:
    if not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("power percent must be in (0, 100]")
    uuid, current, default, minimum = query_power(gpu)
    # Never raise a lower limit installed by another operator.
    watts = math.floor(min(current, default * percent / 100))
    if watts < minimum:
        raise ValueError(f"requested {watts} W is below the GPU minimum {minimum} W")
    return PowerPlan(uuid, current, default, watts, percent)


def verify_limit(gpu: str, expected: float) -> None:
    _, current, _, _ = query_power(gpu)
    if abs(current - expected) > 0.1:
        raise RuntimeError(f"GPU cap is {current} W, expected {expected} W; refusing CUDA work")


def service_command(plan: PowerPlan, command: list[str], cwd: Path,
                    unit: str = "morph-huginn-capped") -> list[str]:
    if os.geteuid() == 0:
        raise RuntimeError("invoke this launcher as your normal user, not through sudo")
    if not re.fullmatch(r"morph-huginn-capped(?:-[a-zA-Z0-9-]+)?", unit):
        raise ValueError("invalid capped service name")
    smi = shutil.which("nvidia-smi")
    systemd = shutil.which("systemd-run")
    pkexec = shutil.which("pkexec")
    if not all((smi, systemd, pkexec)):
        raise RuntimeError("nvidia-smi, systemd-run and pkexec are required")
    user = pwd.getpwuid(os.getuid())
    return [
        pkexec, "--disable-internal-agent", systemd, "--expand-environment=no", f"--unit={unit}",
        "--description=Huginn with temporary GPU power cap and automatic restoration",
        "--property=Type=exec", "--property=Restart=no", "--property=RemainAfterExit=no",
        f"--property=User={user.pw_uid}", f"--property=Group={user.pw_gid}",
        f"--property=WorkingDirectory={cwd}", "--property=KillMode=control-group",
        "--property=TimeoutStartSec=60", "--property=TimeoutStopSec=20",
        f"--property=ExecStartPre=+{smi} -i {plan.uuid} -pl {plan.limit_watts}",
        f"--property=ExecStopPost=+{smi} -i {plan.uuid} -pl {plan.previous_watts:g}",
        "--", *command,
    ]


def start_capped(plan: PowerPlan, command: list[str], cwd: Path, output: Path,
                 unit: str = "morph-huginn-capped") -> None:
    argv = service_command(plan, command, cwd, unit)
    # The fixed system-service name prevents two copies from owning the same cap.
    record = {**asdict(plan), "unit": unit, "command": argv,
              "restoration": "systemd ExecStopPost, including failed start and SIGKILL"}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "power_limit.json"
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
    print(f"Authorize the desktop dialog: {plan.limit_watts} W during this run; "
          f"systemd restores {plan.previous_watts:g} W afterward.", flush=True)
    try:
        subprocess.run(argv, check=True)
    except BaseException:
        # Preserve the request, including failed/cancelled authorization.
        path.rename(output / f"power_limit-unstarted-{os.getpid()}.json")
        raise
