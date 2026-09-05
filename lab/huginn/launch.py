"""Run a Huginn command with persistent logs and its real child exit status."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lab.huginn.power_limit import make_plan, start_capped, verify_limit


def prepare_output(output: Path, resume: bool) -> None:
    """Preserve failed-launch records while keeping completed depth checkpoints."""
    output.mkdir(parents=True, exist_ok=True)
    status = output / "process.json"
    if resume and not (output / "sweep_huginn.json").is_file():
        raise FileNotFoundError("resume requires an existing sweep checkpoint")
    if status.exists():
        if not resume:
            raise FileExistsError(f"Use a new launch directory: {status}")
        previous = json.loads(status.read_text())
        pid = previous.get("pid")
        if pid and Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"previous evaluator PID {pid} still exists; refusing duplicate")
        archive = output / "launch_history" / str(time.time_ns())
        archive.mkdir(parents=True)
        status.rename(archive / status.name)
        for name in ("run.log", "power_limit.json"):
            path = output / name
            if path.exists():
                path.rename(archive / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--power-percent", type=float)
    parser.add_argument("--auth", choices=("sudo", "pkexec"), default="sudo")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--expected-power-watts", type=float)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    resume = "resume=true" in args.overrides
    if args.power_percent is not None:
        prepare_output(output, resume)
        plan = make_plan(args.gpu, args.power_percent)
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                   "--gpu", plan.uuid, "--expected-power-watts", str(plan.limit_watts), *args.overrides]
        start_capped(plan, command, Path(__file__).resolve().parents[2], output, auth=args.auth)
        return
    if args.expected_power_watts is not None:
        verify_limit(args.gpu, args.expected_power_watts)
    prepare_output(output, resume)
    status = output / "process.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HYDRA_FULL_ERROR="1")
    if args.expected_power_watts is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = [sys.executable, "-u", str(Path(__file__).with_name("huginn_depth_sweep.py")),
               f"output={output}", *args.overrides]
    with (output / "run.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        record = {"supervisor_pid": os.getpid(), "pid": process.pid, "command": command,
                  "started": time.time(), "status": "running",
                  "gpu": args.gpu, "power_limit_watts": args.expected_power_watts}
        status.write_text(json.dumps(record, indent=2))
        def terminate(signum, frame):
            if process.poll() is None:
                process.send_signal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, terminate)
        result = process.wait()
    record.update(exit_code=result, ended=time.time(), status="complete" if result == 0 else "failed")
    temporary = status.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(status)
    sys.exit(result if result >= 0 else 128 - result)


if __name__ == "__main__":
    main()
