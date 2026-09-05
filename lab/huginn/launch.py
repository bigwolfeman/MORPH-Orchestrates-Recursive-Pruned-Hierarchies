"""Run a Huginn command with persistent logs and its real child exit status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = output / "process.json"
    if status.exists():
        raise FileExistsError(f"Use a new launch directory: {status}")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HYDRA_FULL_ERROR="1")
    command = [sys.executable, "-u", str(Path(__file__).with_name("huginn_depth_sweep.py")),
               f"output={output}", *args.overrides]
    with (output / "run.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        record = {"supervisor_pid": os.getpid(), "pid": process.pid, "command": command,
                  "started": time.time(), "status": "running"}
        status.write_text(json.dumps(record, indent=2))
        result = process.wait()
    record.update(exit_code=result, ended=time.time(), status="complete" if result == 0 else "failed")
    temporary = status.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(status)
    sys.exit(result if result >= 0 else 128 - result)


if __name__ == "__main__":
    main()
