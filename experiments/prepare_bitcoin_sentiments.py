from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure local imports work when executed as a script.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.preprocessing import preprocess_text


def score_to_label(score: float, neutral_band: float = 0.05) -> int:
    """
    Map continuous sentiment score to discrete label ids:
    - score > neutral_band: Bullish (2)
    - score < -neutral_band: Bearish (0)
    - otherwise: Neutral (1)
    """
    if score > neutral_band:
        return 2
    if score < -neutral_band:
        return 0
    return 1


def load_and_prepare(input_path: Path, neutral_band: float) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if not {"Short Description", "Accurate Sentiments"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'Short Description' and 'Accurate Sentiments' columns.")

    df = df.rename(columns={"Short Description": "text", "Accurate Sentiments": "score"})
    df["label"] = df["score"].apply(lambda s: score_to_label(float(s), neutral_band))
    df["text"] = df["text"].astype(str).apply(preprocess_text)
    return df[["text", "label"]]


def save_splits(df: pd.DataFrame, output_dir: Path, train_size: float = 0.8) -> Tuple[Path, Path]:
    train_df, valid_df = train_test_split(df, train_size=train_size, random_state=42, stratify=df["label"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "bitcoin_sent_train.csv"
    valid_path = output_dir / "bitcoin_sent_valid.csv"
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    return train_path, valid_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare bitcoin_sentiments_21_24.csv into model-ready train/valid splits."
    )
    parser.add_argument("--input", type=Path, default=Path("bitcoin_sentiments_21_24.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--neutral_band", type=float, default=0.05, help="|score| below this is Neutral.")
    parser.add_argument("--train_size", type=float, default=0.8)
    args = parser.parse_args()

    df = load_and_prepare(args.input, args.neutral_band)
    train_path, valid_path = save_splits(df, args.output_dir, train_size=args.train_size)
    print(f"Wrote train split to {train_path}")
    print(f"Wrote valid split to {valid_path}")


if __name__ == "__main__":
    main()
