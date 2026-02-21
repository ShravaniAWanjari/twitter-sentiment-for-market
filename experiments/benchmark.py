from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.metrics import compute_classification_metrics
from core.preprocessing import align_label, tokenize_with_preprocessing
from model_factory import available_models, load_model


class BenchmarkRunner:
    def __init__(
        self,
        valid_path: Path,
        slang_path: Optional[Path] = None,
        *,
        batch_size: int = 32,
        max_length: int = 256,
        latency_samples: int = 64,
        latency_repeats: int = 20,
    ) -> None:
        self.valid_df = pd.read_csv(valid_path)
        self.slang_df = pd.read_csv(slang_path) if slang_path and slang_path.exists() else None
        self.batch_size = batch_size
        self.max_length = max_length
        self.latency_samples = latency_samples
        self.latency_repeats = latency_repeats
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _tokenize_batch(self, model, texts: List[str]):
        encoded = tokenize_with_preprocessing(
            model.tokenizer,
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _predict(self, model, df: pd.DataFrame):
        model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        for start in tqdm(range(0, len(df), self.batch_size), desc="eval", leave=False):
            batch = df.iloc[start : start + self.batch_size]
            labels = [align_label(lbl) for lbl in batch["label"]]
            inputs = self._tokenize_batch(model, batch["text"].astype(str).tolist())
            with torch.no_grad():
                outputs = model(**inputs)
                preds = outputs["logits"].argmax(dim=-1).detach().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
        return all_labels, all_preds

    def _measure_latency(self, model, sample_texts: List[str]) -> float:
        model.eval()
        inputs = self._tokenize_batch(model, sample_texts)
        batch_size = inputs["input_ids"].size(0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            # Warmup to stabilize kernels.
            for _ in range(3):
                model(**inputs)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            timings: List[float] = []
            for _ in range(self.latency_repeats):
                start_event.record()
                model(**inputs)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
            avg_ms = sum(timings) / len(timings)
        else:
            durations: List[float] = []
            for _ in range(self.latency_repeats):
                t0 = time.perf_counter()
                model(**inputs)
                t1 = time.perf_counter()
                durations.append((t1 - t0) * 1000)
            avg_ms = sum(durations) / len(durations)

        return avg_ms / batch_size

    def evaluate_model(self, name: str) -> Dict[str, float]:
        model_kwargs = {"device": self.device, "max_length": self.max_length}
        if name.lower() == "modernbert":
            model_kwargs["use_flash_attention"] = self.device.type == "cuda"
        model = load_model(name, **model_kwargs)
        model.to(self.device)

        labels, preds = self._predict(model, self.valid_df)
        metrics = compute_classification_metrics(labels, preds)

        latency_texts = self.valid_df.sample(
            min(self.latency_samples, len(self.valid_df)), random_state=42
        )["text"].astype(str).tolist()
        latency_ms_per_tweet = self._measure_latency(model, latency_texts)

        slang_accuracy = None
        if self.slang_df is not None and len(self.slang_df) > 0:
            slang_labels, slang_preds = self._predict(model, self.slang_df)
            slang_metrics = compute_classification_metrics(slang_labels, slang_preds)
            slang_accuracy = slang_metrics["accuracy"]

        return {
            "model": name,
            **metrics,
            "latency_ms_per_tweet": latency_ms_per_tweet,
            "slang_accuracy": slang_accuracy,
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark crypto sentiment transformers.")
    parser.add_argument("--valid_path", type=Path, default=Path("dataset/sent_valid.csv"))
    parser.add_argument(
        "--slang_path",
        type=Path,
        default=Path("dataset/sent_slang.csv"),
        help="Slang-heavy test set; optional but recommended.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--latency_samples", type=int, default=64)
    parser.add_argument("--latency_repeats", type=int, default=20)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["modernbert", "cryptobert", "finbert", "deberta-v3"],
        help=f"Subset of models to benchmark. Available: {available_models()}",
    )
    parser.add_argument(
        "--output_csv", type=Path, default=Path("benchmark_results.csv")
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(
        args.valid_path,
        slang_path=args.slang_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        latency_samples=args.latency_samples,
        latency_repeats=args.latency_repeats,
    )

    all_results: List[Dict[str, float]] = []
    for name in args.models:
        print(f"Benchmarking {name} on {runner.device}...")
        all_results.append(runner.evaluate_model(name))

    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
