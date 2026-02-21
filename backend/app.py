from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import sys
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).info("Python executable: %s", sys.executable)
logging.getLogger(__name__).info("sys.path: %s", sys.path)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

try:
    from .job_manager import JobManager
except ImportError:
    from job_manager import JobManager

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
