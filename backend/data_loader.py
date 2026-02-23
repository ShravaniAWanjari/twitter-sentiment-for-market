import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Centralized utility for fetching and converting market tick data.
    """
    BASE_URL = "http://13.231.157.215:8004"
    ENDPOINT_PATH = "/market_data/api/download-csv"
    REQUEST_TIMEOUT = 300

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data_range(
        self, 
        symbol: str = "BTCUSDT", 
        from_date: str = "2025-01-01", 
        to_date: str = "2025-01-31"
    ) -> List[Path]:
        """
        Download and process data for a given range.
        Returns a list of paths to the processed CSV files.
        """
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = datetime.strptime(to_date, "%Y-%m-%d").date()
        
        all_paths = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            path = self.download_and_process_day(symbol, date_str)
            if path:
                all_paths.append(path)
            current += timedelta(days=1)
            
        return all_paths

    def download_and_process_day(self, symbol: str, date: str) -> Optional[Path]:
        """
        Download orderbook snapshot for a single day, convert to flat format, and save.
        """
        filename = f"{symbol}_{date}.csv"
        final_path = self.data_dir / filename
        
        if final_path.exists():
            logger.info(f"Data for {date} already exists at {final_path}")
            return final_path

        payload = {
            "exchange_name": "binance",
            "instrument_type": "spot",
            "data_type": "book_snapshot_5",
            "symbol": symbol,
            "from_date": date,
            "to_date": date,
        }

        try:
            logger.info(f"Downloading data for {date}...")
            response = requests.post(
                f"{self.BASE_URL}{self.ENDPOINT_PATH}",
                json=payload,
                stream=True,
                timeout=self.REQUEST_TIMEOUT
            )

            if response.status_code != 200:
                logger.error(f"Failed to download data for {date}: {response.status_code}")
                return None

            temp_gz = self.data_dir / f"temp_{date}.csv.gz"
            with open(temp_gz, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Process and convert
            logger.info(f"Processing data for {date}...")
            df = pd.read_csv(temp_gz, compression='gzip')
            
            required_cols = ['timestamp', 'bids[0].price', 'asks[0].price']
            if all(col in df.columns for col in required_cols):
                # Clean and filter
                df = df[required_cols].copy()
                df = df[
                    (df['bids[0].price'] > 0) & 
                    (df['asks[0].price'] > 0) & 
                    (df['asks[0].price'] > df['bids[0].price'])
                ]
                df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                
                # Save flat
                df.to_csv(final_path, index=False)
                temp_gz.unlink() # Cleanup
                logger.info(f"Successfully saved {len(df)} rows to {final_path}")
                return final_path
            else:
                logger.warning(f"Missing columns for {date}, saving raw...")
                temp_gz.rename(final_path)
                return final_path

        except Exception as e:
            logger.error(f"Error processing {date}: {e}")
            return None

    def get_combined_df(self, paths: List[Path]) -> pd.DataFrame:
        """
        Helper to load and combine multiple processed CSVs.
        """
        dfs = []
        for p in paths:
            if p.exists():
                dfs.append(pd.read_csv(p))
        
        if not dfs:
            return pd.DataFrame()
            
        combined = pd.concat(dfs, ignore_index=True)
        # Convert timestamp to datetime
        combined['Date'] = pd.to_datetime(combined['timestamp'], unit='ms')
        # Use mid-price for Close compatibility
        combined['Close'] = (combined['bids[0].price'] + combined['asks[0].price']) / 2
        return combined.sort_values('Date')
