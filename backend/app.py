from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import sys
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).info("Python executable: %s", sys.executable)
logging.getLogger(__name__).info("sys.path: %s", sys.path)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import os

# Ensure backend directory is in path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from backend.job_manager import JobManager
    from backend.backtest_engine import BacktestEngine
    from backend.analysis_engine import AnalysisEngine
except ImportError:
    from job_manager import JobManager
    from backtest_engine import BacktestEngine
    from analysis_engine import AnalysisEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DATASET_DIR = REPO_ROOT / "dataset"

app = FastAPI(title="Capstone Research Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager = JobManager(REPO_ROOT)
backtest_engine = BacktestEngine(REPO_ROOT)
analysis_engine = AnalysisEngine(REPO_ROOT)


@app.get("/api/config")
def get_config() -> Dict[str, object]:
    return {
        "models": ["modernbert", "cryptobert", "finbert", "bert-base", "roberta-base"],
        "datasets": {
            "train": str(DATASET_DIR / "bitcoin_sent_train.csv"),
            "valid": str(DATASET_DIR / "bitcoin_sent_valid.csv"),
        },
    }


@app.post("/api/train")
def start_training(payload: Dict[str, List[str]]) -> Dict[str, str]:
    models = payload.get("models", [])
    if not models:
        raise HTTPException(status_code=400, detail="No models provided.")
    try:
        job_manager.start_training(models)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "started"}


@app.post("/api/clear-session")
def clear_session() -> Dict[str, str]:
    try:
        job_manager.clear_session()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "cleared"}


@app.get("/api/status")
def get_status() -> Dict[str, object]:
    return job_manager.get_state()


@app.get("/api/benchmark")
def get_benchmark() -> Dict[str, object]:
    path = REPO_ROOT / "benchmark_results.csv"
    if not path.exists():
        return {"rows": []}
    import pandas as pd

    df = pd.read_csv(path)
    return {"rows": df.to_dict(orient="records")}


@app.get("/api/errors/{name}")
def get_error_csv(name: str) -> Dict[str, object]:
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        return {"rows": []}
    import pandas as pd

    df = pd.read_csv(path)
    return {"rows": df.to_dict(orient="records")}


@app.get("/api/images")
def list_images() -> Dict[str, List[str]]:
    if not RESULTS_DIR.exists():
        return {"images": []}
    images = [p.name for p in RESULTS_DIR.glob("*.png")]
    return {"images": images}


@app.get("/api/images/{name}")
def get_image(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path)


@app.post("/api/backtest")
def run_backtest(payload: Dict[str, Any]):
    return backtest_engine.run_backtest(
        model_name=payload.get("model", "modernbert"),
        strategy_name=payload.get("strategy", "Momentum"),
        sentiment_threshold=payload.get("sentiment_threshold", payload.get("threshold", 0.5))
    )


@app.get("/api/backtest/latest")
def get_latest_backtest():
    run = backtest_engine.get_latest_run()
    if not run:
        raise HTTPException(status_code=404, detail="No backtest run found.")
    return run


@app.post("/api/analyze")
def analyze_headlines(payload: Dict[str, Any]):
    headlines = payload.get("headlines")
    if not headlines:
        # Fallback to demo headlines
        headlines = [
            "Bitcoin breaks $100k as institutional adoption surges",
            "SEC delays decision on spot Ethereum ETF, market nervous",
            "China declares all crypto transactions illegal in major crackdown"
        ]
    return analysis_engine.analyze_headlines(
        model_name=payload.get("model", "modernbert"),
        headlines=headlines
    )


@app.post("/api/chat")
def grounded_chat(payload: Dict[str, Any]):
    run = backtest_engine.get_latest_run()
    if not run:
        return {"response": "I don't have any backtest data yet. Please run a backtest first."}
    response = analysis_engine.grounded_chat(run, payload.get("query", ""))
    return {"response": response}


@app.get("/api/pdf")
def download_pdf():
    run = backtest_engine.get_latest_run()
    if not run:
        raise HTTPException(status_code=404, detail="No run data to export.")
    pdf_path = analysis_engine.generate_pdf_report(run)
    return FileResponse(pdf_path, filename="Analysis_Report.pdf")
