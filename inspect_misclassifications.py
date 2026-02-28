import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from core.preprocessing import LABEL_ID2STR, align_label, preprocess_text
from model_factory import load_model, available_models


URL_PATTERN = re.compile(r"http[s]?://\\S+")
HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
SLANG_TERMS = {
    "hodl",
    "rekt",
    "moon",
    "mooning",
    "pump",
    "dump",
    "bulls",
    "bears",
    "ath",
    "fomo",
    "fud",
}


def batch_iter(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def classify_length(word_count):
    if word_count <= 5:
        return "0-5"
    if word_count <= 10:
        return "6-10"
    if word_count <= 20:
        return "11-20"
    if word_count <= 40:
        return "21-40"
    return "41+"


def main():
    parser = argparse.ArgumentParser(description="Inspect misclassified samples.")
    parser.add_argument("--model", type=str, default="modernbert", help=f"Available: {available_models()}")
    parser.add_argument("--model_dir", type=Path, default=Path("experiments/modernbert_runs/final_model"))
    parser.add_argument("--data_path", type=Path, default=Path("dataset/bitcoin_sent_valid.csv"))
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_csv", type=Path, default=Path("results/misclassified_samples.csv"))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    texts = df[args.text_col].astype(str).tolist()
    raw_labels = df[args.label_col].tolist()
    labels = [align_label(lbl) for lbl in raw_labels]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model weights for {args.model}...", flush=True)
    # Use load_model for consistency and architecture handling
    model_wrapper = load_model(
        args.model, 
        model_name=str(args.model_dir),
        device=device,
        max_length=args.max_length,
        trust_remote_code=True
    )
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    model.eval()

    preds = []
    probs = []
    with torch.no_grad():
        for batch in batch_iter(texts, args.batch_size):
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            scores = torch.softmax(logits, dim=-1)
            preds.extend(scores.argmax(dim=-1).cpu().tolist())
            probs.extend(scores.cpu().tolist())

    all_records = []
    error_records = []
    for text, true_label, pred, prob_vec in zip(texts, labels, preds, probs):
        cleaned = preprocess_text(text)
        words = cleaned.split()
        word_count = len(words)
        char_count = len(cleaned)
        has_url = bool(URL_PATTERN.search(text))
        has_handle = bool(HANDLE_PATTERN.search(text))
        has_slang = any(term in cleaned.split() for term in SLANG_TERMS)
        length_bucket = classify_length(word_count)
        topk = sorted(enumerate(prob_vec), key=lambda x: x[1], reverse=True)[: args.top_k]
        topk_str = ", ".join(f"{LABEL_ID2STR[i]}:{p:.4f}" for i, p in topk)
        pred_confidence = float(max(prob_vec))
        record = {
            "text": text,
            "clean_text": cleaned,
            "true_label": int(true_label),
            "pred_label": int(pred),
            "true_label_str": LABEL_ID2STR.get(int(true_label), str(true_label)),
            "pred_label_str": LABEL_ID2STR.get(int(pred), str(pred)),
            "topk_probs": topk_str,
            "pred_confidence": pred_confidence,
            "word_count": word_count,
            "char_count": char_count,
            "length_bucket": length_bucket,
            "has_url": int(has_url),
            "has_handle": int(has_handle),
            "has_slang": int(has_slang),
            "is_error": int(int(true_label) != int(pred)),
        }
        all_records.append(record)
        if int(true_label) != int(pred):
            error_records.append(record)

    all_df = pd.DataFrame(all_records)
    error_df = pd.DataFrame(error_records).sort_values("pred_confidence", ascending=False)
    error_df.to_csv(args.output_dir / "misclassified_samples.csv", index=False)

    summary = {
        "total_samples": len(all_df),
        "total_errors": int(all_df["is_error"].sum()),
        "error_rate": float(all_df["is_error"].mean()),
        "accuracy": float(1.0 - all_df["is_error"].mean()),
        "avg_confidence": float(all_df["pred_confidence"].mean()),
        "avg_confidence_errors": float(all_df.loc[all_df["is_error"] == 1, "pred_confidence"].mean()),
    }
    summary_df = pd.DataFrame([{"key": k, "value": v} for k, v in summary.items()])
    summary_df.to_csv(args.output_dir / "error_summary.csv", index=False)

    confusion = (
        all_df.groupby(["true_label", "pred_label", "true_label_str", "pred_label_str"])["is_error"]
        .count()
        .reset_index()
        .rename(columns={"is_error": "count"})
    )
    confusion["is_error"] = (confusion["true_label"] != confusion["pred_label"]).astype(int)
    confusion.to_csv(args.output_dir / "confusion_pairs.csv", index=False)

    length_order = ["0-5", "6-10", "11-20", "21-40", "41+"]
    all_df["length_bucket"] = pd.Categorical(all_df["length_bucket"], categories=length_order, ordered=True)
    length_stats = (
        all_df.groupby("length_bucket")["is_error"]
        .agg(total="count", errors="sum")
        .reset_index()
        .sort_values("length_bucket")
    )
    length_stats["error_rate"] = length_stats["errors"] / length_stats["total"]
    length_stats.to_csv(args.output_dir / "error_by_length.csv", index=False)

    signal_rows = []
    for signal in ("has_url", "has_handle", "has_slang"):
        for value in (0, 1):
            subset = all_df[all_df[signal] == value]
            total = len(subset)
            errors = int(subset["is_error"].sum())
            signal_rows.append(
                {
                    "signal": signal,
                    "value": value,
                    "total": total,
                    "errors": errors,
                    "error_rate": (errors / total) if total else 0.0,
                }
            )
    signal_df = pd.DataFrame(signal_rows)
    signal_df.to_csv(args.output_dir / "error_by_signal.csv", index=False)

    confidence_bins = [0.0, 0.4, 0.6, 0.8, 0.9, 1.0001]
    confidence_labels = ["0.0-0.4", "0.4-0.6", "0.6-0.8", "0.8-0.9", "0.9-1.0"]
    all_df["confidence_bucket"] = pd.cut(
        all_df["pred_confidence"],
        bins=confidence_bins,
        labels=confidence_labels,
        include_lowest=True,
        right=False,
    )
    conf_stats = (
        all_df.groupby("confidence_bucket")["is_error"]
        .agg(total="count", errors="sum")
        .reset_index()
    )
    conf_stats["error_rate"] = conf_stats["errors"] / conf_stats["total"]
    conf_stats["accuracy"] = 1.0 - conf_stats["error_rate"]
    conf_stats["avg_confidence"] = (
        all_df.groupby("confidence_bucket")["pred_confidence"].mean().reset_index()["pred_confidence"]
    )
    conf_stats.to_csv(args.output_dir / "error_by_confidence.csv", index=False)

    pairs_dir = args.output_dir / "misclassified_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for (true_label, pred_label), subset in error_df.groupby(["true_label", "pred_label"]):
        filename = f"true_{true_label}_pred_{pred_label}.csv"
        file_path = pairs_dir / filename
        subset.to_csv(file_path, index=False)
        relative_path = file_path.relative_to(args.output_dir).as_posix()
        index_rows.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "true_label_str": LABEL_ID2STR.get(int(true_label), str(true_label)),
                "pred_label_str": LABEL_ID2STR.get(int(pred_label), str(pred_label)),
                "count": len(subset),
                "csv_path": relative_path,
            }
        )
    pd.DataFrame(index_rows).to_csv(pairs_dir / "index.csv", index=False)

    print(f"Wrote {len(error_df)} misclassified samples to {args.output_csv}", flush=True)
    print(f"Wrote analysis CSVs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
