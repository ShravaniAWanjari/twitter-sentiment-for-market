from pathlib import Path
import sys
import os

# Ensure backend is in path
sys.path.append(os.getcwd())

from backend.data_loader import DataLoader

def pre_process():
    repo_root = Path(os.getcwd())
    data_dir = repo_root / "dataset" / "btc_data"
    loader = DataLoader(data_dir)
    
    # Process all temp files we see
    temp_files = list(data_dir.glob("temp_*.csv.gz"))
    print(f"Found {len(temp_files)} compressed files. Processing...")
    
    for tf in temp_files:
        # Extract date from temp_YYYY-MM-DD.csv.gz
        date_str = tf.name.split("_")[1].split(".")[0]
        loader.download_and_process_day("BTCUSDT", date_str)

if __name__ == "__main__":
    pre_process()
