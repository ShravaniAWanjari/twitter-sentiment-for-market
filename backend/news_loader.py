import os
import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class NewsLoader:
    def __init__(self):
        self.api_key = os.getenv("NEWSDATAIO_API_KEY")
        self.base_url = "https://newsdata.io/api/1/crypto"
        if not self.api_key:
            logger.warning("NEWSDATAIO_API_KEY not found in .env")

    def fetch_news(
        self, 
        coin: str = "btc", 
        language: str = "en", 
        from_date: Optional[str] = None, 
        to_date: Optional[str] = None,
        size: int = 10
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        params = {
            "apikey": self.api_key,
            "coin": coin,
            "language": language,
            "size": size
        }
        
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        try:
            logger.info(f"Fetching news from NewsData.io: {params}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success":
                return data.get("results", [])
            else:
                logger.error(f"NewsData.io error: {data.get('message')}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return []

    def fetch_historical_news(self, date_str: str) -> List[Dict[str, Any]]:
        # NewsData.io uses YYYY-MM-DD format for from_date/to_date
        # We fetch for a single day by setting from_date and to_date
        return self.fetch_news(from_date=date_str, to_date=date_str, size=10)

    def fetch_latest_news(self) -> List[Dict[str, Any]]:
        return self.fetch_news(size=10)
