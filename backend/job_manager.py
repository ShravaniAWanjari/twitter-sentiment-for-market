from __future__ import annotations

import threading
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Dict, List, Optional


@dataclass
class JobState:
    job_id: str
    status: str = "idle"  # idle, running, completed, failed
    logs: List[str] = field(default_factory=list)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0


class JobManager:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.state = JobState(job_id="default")
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def _log(self, message: str) -> None:
        with self._lock:
            self.state.logs.append(message)

    def _run_cmd(self, cmd: List[str]) -> int:
        self._log(f"$ {' '.join(cmd)}")
        process = Popen(cmd, cwd=str(self.repo_root), stdout=PIPE, stderr=STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            self._log(line.rstrip())
        return process.wait()

    def start_training(self, models: List[str]) -> None:
        with self._lock:
            if self.state.status == "running":
                raise RuntimeError("Training already running.")
            self.state = JobState(job_id="default", status="running", started_at=time.time())
            self.state.total_steps = len(models) * 2 + 1
            self.state.current_step = 0

        def _worker():
            try:
                for model in models:
                    cmd = [
                        sys.executable,
                        "experiments/train_model.py",
                        "--model",
                        model,
                        "--output_dir",
                        "experiments/runs",
                        "--preflight",
                    ]
                    code = self._run_cmd(cmd)
                    if code != 0:
                        raise RuntimeError(f"Training failed for {model} (exit code {code}).")
                    with self._lock:
                        self.state.current_step += 1

                bench_cmd = [
                    sys.executable,
                    "experiments/benchmark.py",
                    "--valid_path",
                    "dataset/bitcoin_sent_valid.csv",
                    "--models",
                ] + models
                code = self._run_cmd(bench_cmd)
                if code != 0:
                    raise RuntimeError(f"Benchmark failed (exit code {code}).")
                with self._lock:
                    self.state.current_step += 1

                with self._lock:
                    self.state.status = "completed"
                    self.state.finished_at = time.time()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.state.status = "failed"
                    self.state.error = str(exc)
                    self.state.finished_at = time.time()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            progress = (
                min(1.0, self.state.current_step / self.state.total_steps)
                if self.state.total_steps > 0
                else 0.0
            )
            return {
                "job_id": self.state.job_id,
                "status": self.state.status,
                "logs": self.state.logs[-500:],
                "started_at": self.state.started_at,
                "finished_at": self.state.finished_at,
                "error": self.state.error,
                "progress": progress,
                "current_step": self.state.current_step,
                "total_steps": self.state.total_steps,
            }
