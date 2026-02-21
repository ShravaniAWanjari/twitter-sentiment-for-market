from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def plot_accuracy_vs_latency(
    csv_path: Path = Path("benchmark_results.csv"),
    output_path: Path = Path("results/accuracy_vs_latency.png"),
) -> Path:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("benchmark_results.csv is empty; run experiments/benchmark.py first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["latency_ms_per_tweet"], df["accuracy"], c="tab:blue")
    for _, row in df.iterrows():
        plt.annotate(row["model"], (row["latency_ms_per_tweet"], row["accuracy"]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Latency (ms per tweet)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Latency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_f1_vs_latency(
    csv_path: Path = Path("benchmark_results.csv"),
    output_path: Path = Path("results/f1_vs_latency.png"),
) -> Path:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("benchmark_results.csv is empty; run experiments/benchmark.py first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["latency_ms_per_tweet"], df["f1_macro"], c="tab:orange")
    for _, row in df.iterrows():
        plt.annotate(row["model"], (row["latency_ms_per_tweet"], row["f1_macro"]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Latency (ms per tweet)")
    plt.ylabel("Macro F1")
    plt.title("Macro F1 vs Latency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


if __name__ == "__main__":
    acc_path = plot_accuracy_vs_latency()
    f1_path = plot_f1_vs_latency()
    print(f"Saved plots to {acc_path} and {f1_path}")
