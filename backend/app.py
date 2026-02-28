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


@app.get("/api/models/trained")
def get_trained_models() -> Dict[str, List[Dict[str, Any]]]:
    status_list = []
    models = ["modernbert", "cryptobert", "finbert", "bert-base", "roberta-base"]
    
    # Load benchmarks to check for has_benchmark
    existing_bench = []
    bench_path = REPO_ROOT / "benchmark_results.csv"
    if bench_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(bench_path)
            existing_bench = df["model"].tolist() if "model" in df.columns else []
        except:
            pass

    for m in models:
        runs_dir = REPO_ROOT / "experiments" / "runs" / m / "final_model"
        analysis_path = REPO_ROOT / "results" / m / "error_summary.csv"
        
        status_list.append({
            "model": m,
            "has_weights": runs_dir.exists(),
            "has_benchmark": m in existing_bench,
            "has_analysis": analysis_path.exists()
        })
    return {"models": status_list}


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
def clear_session(payload: Dict[str, Any] = None) -> Dict[str, str]:
    clear_models = (payload or {}).get("clear_models", False)
    try:
        job_manager.clear_session(clear_models=clear_models)
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


@app.get("/api/errors/summaries/all")
def get_all_summaries() -> Dict[str, Any]:
    summaries = {}
    if not RESULTS_DIR.exists():
        return summaries
    
    # Iterate over subdirectories in results/
    for model_dir in RESULTS_DIR.iterdir():
        if model_dir.is_dir():
            summary_path = model_dir / "error_summary.csv"
            if summary_path.exists():
                import pandas as pd
                df = pd.read_csv(summary_path)
                summaries[model_dir.name] = df.to_dict(orient="records")
    return summaries


@app.get("/api/errors/{name}")
def get_error_csv(name: str, model: str = None) -> Dict[str, object]:
    if model:
        path = RESULTS_DIR / model / f"{name}.csv"
    else:
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
    try:
        return backtest_engine.run_backtest(
            model_name=payload.get("model", "modernbert"),
            strategy_name=payload.get("strategy", "Momentum"),
            sentiment_threshold=payload.get("sentiment_threshold", payload.get("threshold", 0.5)),
            initial_balance=float(payload.get("initial_balance", 10000.0)),
            risk_per_trade=float(payload.get("risk_per_trade", 0.02))
        )
    except Exception as e:
        logger.error(f"Backtest API error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/latest")
def get_latest_backtest():
    run = backtest_engine.get_latest_run()
    if not run:
        raise HTTPException(status_code=404, detail="No backtest run found.")
    return run


@app.post("/api/analyze")
def analyze_headlines(payload: Dict[str, Any]):
    headlines = payload.get("headlines")
    model_name = payload.get("model", "modernbert")
    if not headlines:
        # If no headlines provided, fetch from NewsData.io (automated mode)
        date_str = payload.get("date") # Optional date for historical news
        return analysis_engine.fetch_and_analyze(model_name, date_str)
        
    return analysis_engine.analyze_headlines(
        model_name=model_name,
        headlines=headlines
    )


@app.get("/api/news/latest")
def get_latest_news():
    return analysis_engine.news_loader.fetch_latest_news()


@app.get("/api/news/bitcoin-headlines")
def get_bitcoin_headlines():
    """Fetch top 5 live Bitcoin headlines from NewsData.io."""
    try:
        raw = analysis_engine.news_loader.fetch_news(coin="btc", size=5)
        headlines = []
        for item in raw:
            headlines.append({
                "title": item.get("title", ""),
                "source": item.get("source_id", item.get("source_name", "Unknown")),
                "pubDate": item.get("pubDate", ""),
                "link": item.get("link", ""),
                "description": (item.get("description", "") or "")[:200],
            })
        return {"headlines": headlines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch headlines: {e}")


@app.post("/api/news/analyze-headline")
def analyze_single_headline(payload: Dict[str, Any]):
    """Analyze a single headline with the best model + full explainability."""
    text = payload.get("text", "").strip()
    model_name = payload.get("model", "modernbert")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in payload.")

    # Step 1: Get sentiment prediction with token-level attention scores
    sentiment_results = analysis_engine.analyze_headlines(model_name, [text])
    sentiment = sentiment_results[0] if sentiment_results else {}

    # Step 2: Get full explainability (occlusion, stability, counterfactual)
    explain = analysis_engine.explain_text(model_id=model_name, text=text)

    return {
        "headline": text,
        "model": model_name,
        "sentiment": sentiment.get("sentiment", "Unknown"),
        "confidence": sentiment.get("confidence", 0),
        "top_tokens": sentiment.get("top_tokens", []),
        "explainability": explain,
    }


@app.post("/api/explain")
def explain_text(payload: Dict[str, Any]):
    model_id = payload.get("model_id") or payload.get("model")
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in payload.")
    return analysis_engine.explain_text(
        model_id=model_id or "modernbert",
        text=text
    )


@app.post("/api/chat")
def grounded_chat(payload: Dict[str, Any]):
    run = backtest_engine.get_latest_run()
    
    # Also load benchmark results for broader context
    bench_data = []
    bench_path = REPO_ROOT / "benchmark_results.csv"
    if bench_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(bench_path)
            bench_data = df.to_dict(orient="records")
        except:
            pass

    if not run and not bench_data:
        return {"response": "I don't have any backtest or benchmark data yet. Please run the training and benchmark pipeline first!"}
    
    response = analysis_engine.grounded_chat(run, payload.get("query", ""), benchmark_data=bench_data)
    return {"response": response}


@app.get("/api/pdf")
def download_pdf():
    run = backtest_engine.get_latest_run()
    if not run:
        raise HTTPException(status_code=404, detail="No run data to export.")
    pdf_path = analysis_engine.generate_pdf_report(run)
    return FileResponse(pdf_path, filename="Analysis_Report.pdf")
