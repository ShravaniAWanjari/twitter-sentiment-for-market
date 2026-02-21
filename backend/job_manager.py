from __future__ import annotations

import threading
import time
import re
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
    current_model: Optional[str] = None
    intra_step_progress: float = 0.0  # 0.0 to 1.0 within a step


class JobManager:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.state = JobState(job_id="default")
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def _log(self, message: str) -> None:
        with self._lock:
            # Simple heuristic for tqdm: lines containing "%|" and ending in "it/s]" or "s/it]"
            if "%|" in message and any(x in message for x in ["it/s", "s/it"]):
                if self.state.logs and "%|" in self.state.logs[-1]:
                    self.state.logs[-1] = message
                    return
            self.state.logs.append(message)

    def _get_python_exe(self) -> str:
        # Favor the local venv if it exists, otherwise use current sys.executable
        venv_exe = self.repo_root / "capstone" / "Scripts" / "python.exe"
        if venv_exe.exists():
            return str(venv_exe)
        return sys.executable

    def _run_cmd(self, cmd: List[str]) -> int:
        self._log(f"$ {' '.join(cmd)}")
        process = Popen(cmd, cwd=str(self.repo_root), stdout=PIPE, stderr=STDOUT, text=True)
        assert process.stdout is not None
        
        # Regex to match tqdm progress like "12%|"
        progress_re = re.compile(r"(\d+)%\|")
        
        for line in process.stdout:
            clean_line = line.rstrip()
            self._log(clean_line)
            
            # Parse progress
            match = progress_re.search(clean_line)
            if match:
                try:
                    pct = int(match.group(1)) / 100.0
                    with self._lock:
                        self.state.intra_step_progress = pct
                except (ValueError, IndexError):
                    pass

        return process.wait()

    def start_training(self, models: List[str]) -> None:
        with self._lock:
            if self.state.status == "running":
                raise RuntimeError("Training already running.")
            self.state = JobState(job_id="default", status="running", started_at=time.time())
            self.state.total_steps = len(models) + 1 + 1  # Train models + Benchmark + Analyze
            self.state.current_step = 0

        def _worker():
            try:
                python_exe = self._get_python_exe()
                for model in models:
                    cmd = [
                        python_exe,
                        "experiments/train_model.py",
                        "--model",
                        model,
                        "--output_dir",
                        "experiments/runs",
                        "--preflight",
                    ]
                    with self._lock:
                        self.state.current_model = model
                        self.state.intra_step_progress = 0.0
                    code = self._run_cmd(cmd)
                    if code != 0:
                        raise RuntimeError(f"Training failed for {model} (exit code {code}).")
                    with self._lock:
                        self.state.current_step += 1
                        self.state.intra_step_progress = 0.0

                bench_cmd = [
                    python_exe,
                    "experiments/benchmark.py",
                    "--valid_path",
                    str(self.repo_root / "dataset" / "bitcoin_sent_valid.csv"),
                    "--models",
                ] + models
                with self._lock:
                    self.state.current_model = "Benchmarking"
                    self.state.intra_step_progress = 0.0
                code = self._run_cmd(bench_cmd)
                if code != 0:
                    raise RuntimeError(f"Benchmark failed (exit code {code}).")
                with self._lock:
                    self.state.current_step += 1
                    self.state.intra_step_progress = 0.0

                # Phase 3: Error Analysis (on the first model for simplicity)
                with self._lock:
                    self.state.current_model = f"Analyzing {models[0]}"
                    self.state.intra_step_progress = 0.0
                
                analyze_cmd = [
                    python_exe,
                    "inspect_misclassifications.py",
                    "--model",
                    models[0],
                    "--model_dir",
                    str(self.repo_root / "experiments" / "runs" / models[0] / "final_model"),
                    "--data_path",
                    str(self.repo_root / "dataset" / "bitcoin_sent_valid.csv"),
                    "--output_dir",
                    "results",
                ]
                code = self._run_cmd(analyze_cmd)
                if code != 0:
                    raise RuntimeError(f"Error analysis failed (exit code {code}).")

                with self._lock:
                    self.state.current_step += 1
                    self.state.intra_step_progress = 0.0
                    self.state.current_model = None

                with self._lock:
                    self.state.status = "completed"
                    self.state.finished_at = time.time()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.state.status = "failed"
                    self.state.error = str(exc)
                    self.state.finished_at = time.time()
                    self.state.current_model = None
                    self.state.intra_step_progress = 0.0

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def clear_session(self) -> None:
        with self._lock:
            if self.state.status == "running":
                raise RuntimeError("Cannot clear session while training is running.")

            # 1. Delete benchmark_results.csv
            bench_path = self.repo_root / "benchmark_results.csv"
            if bench_path.exists():
                bench_path.unlink()

            # 2. Delete all files in results/
            results_dir = self.repo_root / "results"
            if results_dir.exists():
                import shutil
                for item in results_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

            # 3. Reset state
            self.state = JobState(job_id="default")
            self._log("Session cleared.")

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
                "current_model": self.state.current_model,
                "intra_step_progress": self.state.intra_step_progress,
            }
