import sys
import os
import threading
import time
import re
import pandas as pd
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
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = Popen(cmd, cwd=str(self.repo_root), stdout=PIPE, stderr=STDOUT, text=True, env=env)
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
            
            # Determine which models actually need what
            untrained = []
            unbenchmarked = []
            unanalyzed = []

            # Load existing benchmarks to check
            existing_bench = []
            bench_csv = self.repo_root / "benchmark_results.csv"
            if bench_csv.exists():
                try:
                    df = pd.read_csv(bench_csv)
                    existing_bench = df["model"].tolist() if "model" in df.columns else []
                except Exception:
                    pass

            for m in models:
                # 1. Training check
                model_dir = self.repo_root / "experiments" / "runs" / m / "final_model"
                if not model_dir.exists():
                    untrained.append(m)
                    # If we must train, we MUST also re-benchmark and re-analyze to be safe
                    unbenchmarked.append(m)
                    unanalyzed.append(m)
                else:
                    # 2. Benchmark check
                    if m not in existing_bench:
                        unbenchmarked.append(m)
                    
                    # 3. Analysis check
                    analysis_file = self.repo_root / "results" / m / "error_summary.csv"
                    if not analysis_file.exists():
                        unanalyzed.append(m)
            
            self.state = JobState(job_id="default", status="running", started_at=time.time())
            # Total steps: models remaining in each phase
            self.state.total_steps = len(untrained) + len(unbenchmarked) + len(unanalyzed)
            self.state.current_step = 0

        def _worker():
            try:
                python_exe = self._get_python_exe()
                
                # Phase 1: Training
                for model in untrained:
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

                for m in models:
                    if m not in untrained:
                        self._log(f"Reusing saved weights for {m}")

                # Phase 2: Benchmarking (Run per model if needed)
                # Only remove rows for models that need re-benchmarking, keep the rest
                if unbenchmarked:
                    bench_csv = self.repo_root / "benchmark_results.csv"
                    if bench_csv.exists():
                        try:
                            df = pd.read_csv(bench_csv)
                            df = df[~df["model"].isin(unbenchmarked)]
                            if df.empty:
                                bench_csv.unlink()
                            else:
                                df.to_csv(bench_csv, index=False)
                            self._log(f"Cleared benchmark rows for: {unbenchmarked}")
                        except Exception:
                            bench_csv.unlink()

                for model in unbenchmarked:
                    with self._lock:
                        if self.state.status != "running":
                            return
                        self.state.current_model = f"Benchmarking {model}"
                        self.state.intra_step_progress = 0.0
                    
                    bench_cmd = [
                        python_exe,
                        "experiments/benchmark.py",
                        "--valid_path",
                        str(self.repo_root / "dataset" / "bitcoin_sent_valid.csv"),
                        "--models",
                        model,
                        "--append"
                    ]
                    code = self._run_cmd(bench_cmd)
                    if code != 0:
                        self._log(f"Warning: Benchmarking failed for {model} (exit code {code}).")
                    
                    with self._lock:
                        self.state.current_step += 1
                        self.state.intra_step_progress = 0.0

                for m in models:
                    if m not in unbenchmarked:
                        self._log(f"Reusing existing benchmark results for {m}")

                # Phase 3: Error Analysis (Run per model if needed)
                for model in unanalyzed:
                    with self._lock:
                        if self.state.status != "running":
                            return
                        self.state.current_model = f"Analyzing {model}"
                        self.state.intra_step_progress = 0.0
                    
                    # Create model-specific subdirectory
                    model_results_dir = Path("results") / model
                    (self.repo_root / model_results_dir).mkdir(parents=True, exist_ok=True)
                    
                    analyze_cmd = [
                        python_exe,
                        "inspect_misclassifications.py",
                        "--model",
                        model,
                        "--model_dir",
                        str(self.repo_root / "experiments" / "runs" / model / "final_model"),
                        "--data_path",
                        str(self.repo_root / "dataset" / "bitcoin_sent_valid.csv"),
                        "--output_dir",
                        str(model_results_dir),
                    ]
                    code = self._run_cmd(analyze_cmd)
                    if code != 0:
                        self._log(f"Warning: Error analysis failed for {model} (exit code {code}).")

                    with self._lock:
                        self.state.current_step += 1
                        self.state.intra_step_progress = 0.0

                for m in models:
                    if m not in unanalyzed:
                        self._log(f"Reusing existing analysis for {m}")
                
                with self._lock:
                    self.state.current_model = None

                with self._lock:
                    self.state.status = "completed"
                    self.state.finished_at = time.time()
                    if self.state.total_steps > 0:
                        self.state.current_step = self.state.total_steps
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.state.status = "failed"
                    self.state.error = str(exc)
                    self.state.finished_at = time.time()
                    self.state.current_model = None
                    self.state.intra_step_progress = 0.0

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def _rmtree_with_retries(self, path: Path, retries: int = 5, delay: float = 0.5) -> None:
        import shutil
        import stat
        
        def on_error(func, path, exc_info):
            # Try to fix permission issues on Windows
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

        for i in range(retries):
            try:
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path, onerror=on_error)
                return
            except Exception as e:
                if i == retries - 1:
                    self._log(f"Warning: Failed to delete {path} after {retries} attempts: {e}")
                else:
                    time.sleep(delay)

    def clear_session(self, clear_models: bool = False) -> None:
        with self._lock:
            if self.state.status == "running":
                raise RuntimeError("Cannot clear session while training is running.")
            # Set a temporary status to prevent others from starting training
            self.state.status = "clearing"
            self._log("Clearing session artifacts...")

        def _clear_worker():
            try:
                # 1. Delete benchmark_results.csv
                bench_path = self.repo_root / "benchmark_results.csv"
                self._rmtree_with_retries(bench_path)

                # 2. Delete all files in results/
                results_dir = self.repo_root / "results"
                if results_dir.exists():
                    for item in results_dir.iterdir():
                        self._rmtree_with_retries(item)

                # 3. Delete trained models in experiments/runs/ (ONLY IF REQUESTED)
                if clear_models:
                    runs_dir = self.repo_root / "experiments" / "runs"
                    if runs_dir.exists():
                        for item in runs_dir.iterdir():
                            if item.is_dir():
                                self._rmtree_with_retries(item)
            except Exception as e:
                self._log(f"Warning: Error during background cleanup: {e}")
            finally:
                with self._lock:
                    # 4. Reset state to IDLE with a fresh job ID
                    self.state = JobState(job_id=f"default_{int(time.time())}")
                    self._log("Session cleared.")

        threading.Thread(target=_clear_worker, daemon=True).start()

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
