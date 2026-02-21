from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.metrics import compute_classification_metrics
from core.preprocessing import align_label, tokenize_with_preprocessing
from model_factory import load_model


@dataclass
class TrainingConfig:
    train_path: Path
    valid_path: Path
    output_dir: Path
    epochs: int = 2
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 256
    use_flash_attention: bool = True


class TokenizedTweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        texts: List[str] = df["text"].astype(str).tolist()
        labels: List[int] = [align_label(lbl) for lbl in df["label"]]
        encodings = tokenize_with_preprocessing(
            tokenizer,
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_datasets(config: TrainingConfig, tokenizer, max_length: int):
    train_df = pd.read_csv(config.train_path)
    valid_df = pd.read_csv(config.valid_path)
    train_ds = TokenizedTweetDataset(train_df, tokenizer, max_length)
    valid_ds = TokenizedTweetDataset(valid_df, tokenizer, max_length)
    return train_ds, valid_ds


def main():
    parser = argparse.ArgumentParser(description="Train ModernBERT with Flash Attention 2.0")
    parser.add_argument("--train_path", type=Path, default=Path("dataset/sent_train.csv"))
    parser.add_argument("--valid_path", type=Path, default=Path("dataset/sent_valid.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/modernbert_runs"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--no-flash", action="store_true", help="Disable Flash Attention 2.0 even if available.")
    args = parser.parse_args()

    config = TrainingConfig(
        train_path=args.train_path,
        valid_path=args.valid_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        use_flash_attention=not args.no_flash,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_flash = config.use_flash_attention and torch.cuda.is_available()

    model = load_model(
        "modernbert",
        use_flash_attention=use_flash,
        device=device,
        max_length=config.max_length,
    )

    # Enable Flash SDP kernels if supported (RTX 40-series).
    if use_flash:
        try:
            torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
        except Exception:
            pass

    train_ds, valid_ds = build_datasets(config, model.tokenizer, config.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer)

    # Build TrainingArguments defensively to support older transformers without evaluation_strategy.
    import inspect

    arg_options = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1_macro",
        "greater_is_better": True,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
    }
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered_args = {k: v for k, v in arg_options.items() if k in supported}
    if "evaluation_strategy" not in filtered_args:
        for key in ("save_strategy", "load_best_model_at_end", "metric_for_best_model", "greater_is_better"):
            filtered_args.pop(key, None)
    training_args = TrainingArguments(**filtered_args)

    def compute_metrics_fn(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        preds = logits.argmax(axis=-1)
        metrics = compute_classification_metrics(labels, preds)
        # Trainer expects keys prefixed with eval_ at evaluation time.
        return {f"eval_{k}": v for k, v in metrics.items()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    trainer.save_model(str(config.output_dir / "final_model"))
    model.tokenizer.save_pretrained(str(config.output_dir / "final_model"))


if __name__ == "__main__":
    main()
