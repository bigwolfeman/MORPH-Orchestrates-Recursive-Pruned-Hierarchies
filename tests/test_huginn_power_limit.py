import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab.huginn import launch, power_limit as power


def test_percent_uses_default_and_preserves_lower_existing_limit(monkeypatch):
    monkeypatch.setattr(power, "query_power", lambda gpu: ("GPU-abcd", 575., 575., 400.))
    plan = power.make_plan("0", 92)
    assert plan.limit_watts == 529
    assert plan.previous_watts == 575
    monkeypatch.setattr(power, "query_power", lambda gpu: ("GPU-abcd", 450., 575., 400.))
    assert power.make_plan("0", 92).limit_watts == 450
    with pytest.raises(ValueError, match="minimum"):
        power.make_plan("0", 50)
    for percent in (0, -1, 101, float("nan")):
        with pytest.raises(ValueError):
            power.make_plan("0", percent)


def test_system_service_restores_previous_cap_and_runs_model_unprivileged(monkeypatch):
    monkeypatch.setattr(power.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(power.os, "getuid", lambda: 1000)
    monkeypatch.setattr(power.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_uid=1000, pw_gid=1000))
    monkeypatch.setattr(power.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    plan = power.PowerPlan("GPU-abcd", 575, 575, 529, 92)
    child = ["/venv/python", "/repo/launch.py", "resume=true"]
    cmd = power.service_command(plan, child, Path("/repo"))
    assert cmd[:3] == ["/usr/bin/pkexec", "--disable-internal-agent", "/usr/bin/systemd-run"]
    assert "--property=User=1000" in cmd
    assert "--property=Group=1000" in cmd
    assert "--property=KillMode=control-group" in cmd
    assert "--property=ExecStartPre=+/usr/bin/nvidia-smi -i GPU-abcd -pl 529" in cmd
    assert "--property=ExecStopPost=+/usr/bin/nvidia-smi -i GPU-abcd -pl 575" in cmd
    assert cmd[cmd.index("--") + 1:] == child
    assert "--expand-environment=no" in cmd
    assert "--collect" not in cmd


def test_launcher_refuses_work_when_cap_was_not_applied(monkeypatch, tmp_path):
    monkeypatch.setattr(power, "query_power", lambda gpu: ("GPU-abcd", 575., 575., 400.))
    monkeypatch.setattr(launch.sys, "argv", ["launch.py", "--output", str(tmp_path),
                                           "--expected-power-watts", "529"])
    def forbidden(*args, **kwargs):
        pytest.fail("CUDA child must not start without the requested power cap")
    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="refusing CUDA"):
        launch.main()
    assert not (tmp_path / "process.json").exists()


def test_resume_preserves_previous_records_and_completed_depths(tmp_path):
    checkpoint = b'{"completed_depth": 32}'
    (tmp_path / "sweep_huginn.json").write_bytes(checkpoint)
    (tmp_path / "process.json").write_text(json.dumps({"status": "failed", "exit_code": -9}))
    (tmp_path / "run.log").write_text("finished depth 32\npartial depth 48\n")
    launch.prepare_output(tmp_path, resume=True)
    archives = list((tmp_path / "launch_history").iterdir())
    assert len(archives) == 1
    assert json.loads((archives[0] / "process.json").read_text())["exit_code"] == -9
    assert "partial depth 48" in (archives[0] / "run.log").read_text()
    assert (tmp_path / "sweep_huginn.json").read_bytes() == checkpoint
    assert not (tmp_path / "run.log").exists()


def test_resume_refuses_live_pid_and_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        launch.prepare_output(tmp_path, resume=True)
    (tmp_path / "sweep_huginn.json").write_text("{}")
    original = json.dumps({"pid": os.getpid(), "status": "running"})
    (tmp_path / "process.json").write_text(original)
    with pytest.raises(RuntimeError, match="still exists"):
        launch.prepare_output(tmp_path, resume=True)
    assert (tmp_path / "process.json").read_text() == original


def test_setter_failure_is_not_suppressed(monkeypatch):
    def failed(*args, **kwargs):
        raise power.subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(power.subprocess, "run", failed)
    with pytest.raises(power.subprocess.CalledProcessError):
        power.make_plan("0", 92)
