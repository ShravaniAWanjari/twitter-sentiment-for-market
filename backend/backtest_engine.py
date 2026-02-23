from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.data_loader import DataLoader

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results_dir = repo_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.artifact_path = self.results_dir / "RunArtifact.json"
        self.data_loader = DataLoader(repo_root / "dataset" / "btc_data")

    def load_price_data(self) -> pd.DataFrame:
        """
        Load historical BTC data from Jan 2025 using tick data API.
        """
        # We target Jan 2025 as requested
        paths = self.data_loader.fetch_data_range(
            symbol="BTCUSDT",
            from_date="2025-01-01",
            to_date="2025-01-31"
        )
        
        if paths:
            df = self.data_loader.get_combined_df(paths)
            if not df.empty:
                logger.info(f"Loaded {len(df)} ticks from Jan 2025.")
                # Resample to hourly to keep existing backtest logic efficient
                # or we can keep it at tick level, but hourly is safer for the current UI/Performance
                df.set_index('Date', inplace=True)
                resampled = df['Close'].resample('H').last().dropna().reset_index()
                return resampled

        # Fallback to local file or demo data
        price_path = self.repo_root / "dataset" / "btc_historical_price.csv"
        if price_path.exists():
            df = pd.read_csv(price_path)
            df["Date"] = pd.to_datetime(df["Date"])
            return df.sort_values("Date")
        
        logger.info("Tick data failed and local price data not found, generating demo BTC price data.")
        dates = pd.date_range(start="2025-01-01", periods=720, freq="H") # ~1 month
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, size=len(dates))
        price = 95000 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"Date": dates, "Close": price})
        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_momentum(self, prices: pd.Series, window: int = 24) -> pd.Series:
        return prices / prices.shift(window) - 1

    def run_backtest(
        self, 
        model_name: str, 
        strategy_name: str,
        sentiment_threshold: float = 0.5,
        lag_hours: int = 1
    ) -> Dict[str, Any]:
        df = self.load_price_data()
        
        # 1. Base Strategy Signals
        if strategy_name.lower() == "rsi":
            rsi = self.calculate_rsi(df["Close"])
            df["base_signal"] = 0
            df.loc[rsi < 30, "base_signal"] = 1  # Buy
            df.loc[rsi > 70, "base_signal"] = -1 # Sell
        elif strategy_name.lower() == "momentum":
            mom = self.calculate_momentum(df["Close"])
            df["base_signal"] = 0
            df.loc[mom > 0.01, "base_signal"] = 1
            df.loc[mom < -0.01, "base_signal"] = -1
        else:
            df["base_signal"] = 1 # Buy and hold fallback
            
        # 2. Sentiment Gating
        # In a real scenario, we'd join sentiment data here.
        # For MVP, we'll simulate sentiment influence if no sentiment file exists.
        df["sentiment"] = 0.5 # Neutral
        # Look for existing sentiment results (e.g., from modernbert_results.csv)
        sent_path = self.repo_root / f"{model_name}_results.csv"
        if sent_path.exists():
            sent_df = pd.read_csv(sent_path)
            # Basic mapping logic - this is simplified for MVP
            # In a real app, you'd align timestamps accurately.
            # Here we just use a random subset to simulate gating.
            df["sentiment"] = np.random.uniform(0, 1, size=len(df))

        # Gating: If sentiment is below threshold, zero out the signal (defensive)
        df["gated_signal"] = df["base_signal"].copy()
        df.loc[df["sentiment"] < sentiment_threshold, "gated_signal"] = 0
        
        # 3. Calculate Returns
        df["returns"] = df["Close"].pct_change()
        df["baseline_cum"] = (1 + df["base_signal"].shift(lag_hours) * df["returns"]).cumprod()
        df["gated_cum"] = (1 + df["gated_signal"].shift(lag_hours) * df["returns"]).cumprod()
        
        # Fill NaNs
        df = df.fillna(1.0)

        # 4. Metrics
        def get_metrics(cum_series):
            total_ret = cum_series.iloc[-1] - 1
            std = cum_series.pct_change().std()
            sharpe = (cum_series.pct_change().mean() / std) * np.sqrt(365 * 24) if std > 0 else 0
            max_dd = (cum_series / cum_series.cummax() - 1).min()
            # Ensure no NaNs go into the artifact
            return {
                "total_return": float(total_ret) if not np.isnan(total_ret) else 0.0,
                "sharpe_ratio": float(sharpe) if not np.isnan(sharpe) else 0.0,
                "max_drawdown": float(max_dd) if not np.isnan(max_dd) else 0.0
            }

        artifact = {
            "model": model_name,
            "strategy": strategy_name,
            "metrics": {
                "baseline": get_metrics(df["baseline_cum"]),
                "gated": get_metrics(df["gated_cum"])
            },
            "equity_curve": {
                "dates": df["Date"].dt.strftime("%Y-%m-%d %H:%M").tolist()[::10], # Downsample for UI
                "baseline": df["baseline_cum"].tolist()[::10],
                "gated": df["gated_cum"].tolist()[::10]
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }

        with open(self.artifact_path, "w") as f:
            json.dump(artifact, f)
            
        return artifact

    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        if self.artifact_path.exists():
            with open(self.artifact_path, "r") as f:
                return json.load(f)
        return None
