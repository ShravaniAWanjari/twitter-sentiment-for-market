from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATASET_DIR = BASE_DIR / "dataset"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def render_metric_grid(summary: pd.DataFrame) -> None:
    if summary.empty:
        st.info("No summary data available.")
        return
    key_map = {
        "total_samples": "Total samples",
        "total_errors": "Total errors",
        "error_rate": "Error rate",
        "accuracy": "Accuracy",
        "avg_confidence": "Avg confidence",
        "avg_confidence_errors": "Avg confidence (errors)",
    }
    int_keys = {"total_samples", "total_errors"}
    cols = st.columns(3)
    for idx, row in summary.iterrows():
        label = key_map.get(row["key"], row["key"])
        value = row["value"]
        if row["key"] in int_keys:
            display = f"{int(float(value))}"
        elif isinstance(value, (float, int)):
            display = f"{value:.4f}"
        else:
            display = str(value)
        cols[idx % 3].metric(label, display)


def chart_bar(df: pd.DataFrame, x: str, y: str, title: str) -> alt.Chart:
    chart = (
        alt.Chart(df, title=title)
        .mark_bar(color="#6366f1")
        .encode(
            x=alt.X(x, sort=None),
            y=alt.Y(y),
            tooltip=[x, y],
        )
    )
    return chart


def chart_scatter(df: pd.DataFrame, x: str, y: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df, title=title)
        .mark_circle(size=120, color="#818cf8")
        .encode(x=alt.X(x), y=alt.Y(y), tooltip=list(df.columns))
    )


def format_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    fmt = {}
    for col in df.columns:
        if df[col].dtype.kind in {"i", "u"}:
            fmt[col] = "{:d}"
        elif df[col].dtype.kind == "f":
            fmt[col] = "{:.4f}"
    return df.style.format(fmt)


def validate_dataset(df: pd.DataFrame) -> list[str]:
    issues = []
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {', '.join(sorted(missing))}")
        return issues
    if df["text"].isna().any():
        issues.append("Found empty text rows.")
    if df["label"].isna().any():
        issues.append("Found empty labels.")
    return issues


def main() -> None:
    st.set_page_config(page_title="Capstone Dashboard", layout="wide")
    st.title("Capstone Results Dashboard")
    st.caption("Interactive view of training, benchmarking, and error analysis outputs.")

    st.sidebar.header("Data sources")
    st.sidebar.write("Default paths:")
    st.sidebar.code(
        "\n".join(
            [
                "benchmark_results.csv",
                "results/error_summary.csv",
                "results/confusion_pairs.csv",
                "results/error_by_length.csv",
                "results/error_by_signal.csv",
                "results/error_by_confidence.csv",
                "results/misclassified_samples.csv",
                "results/misclassified_pairs/index.csv",
            ]
        )
    )

    tab_portal, tab_overview, tab_benchmark, tab_error, tab_misclass, tab_artifacts = st.tabs(
        [
            "Researcher Portal",
            "Overview",
            "Benchmarks",
            "Error Analysis",
            "Misclassifications",
            "Artifacts",
        ]
    )

    with tab_portal:
        st.subheader("Experiment Setup")
        st.markdown(
            "Choose a model and run experiments against the **preloaded dataset**. "
            "This keeps the pipeline strict and reproducible for comparison."
        )

        available_models = [
            "modernbert",
            "cryptobert",
            "finbert",
            "bert-base",
            "roberta-base",
        ]
        model_choice = st.selectbox("Model", available_models)

        st.markdown("**Dataset**")
        train_path = DATASET_DIR / "bitcoin_sent_train.csv"
        valid_path = DATASET_DIR / "bitcoin_sent_valid.csv"
        st.code(f"{train_path}\n{valid_path}")
        train_df = load_csv(train_path)
        valid_df = load_csv(valid_path)
        if train_df.empty or valid_df.empty:
            st.warning("Train/valid dataset not found. Make sure dataset splits exist.")
        else:
            train_issues = validate_dataset(train_df)
            valid_issues = validate_dataset(valid_df)
            cols = st.columns(2)
            cols[0].metric("Train rows", f"{len(train_df)}")
            cols[1].metric("Valid rows", f"{len(valid_df)}")
            if train_issues or valid_issues:
                st.error("Dataset checks failed:")
                for issue in train_issues + valid_issues:
                    st.write(f"- {issue}")
            else:
                st.success("Dataset checks passed.")

        st.markdown("**Mode**")
        st.info(
            "Current mode supports BERT‑family models via the generic trainer. "
            "Non‑BERT models require an adapter and schema mapping."
        )

        st.markdown("**Run commands**")
        st.code(
            "\n".join(
                [
                    f"python experiments/train_model.py --model {model_choice} --output_dir experiments\\runs",
                    "python experiments/benchmark.py --valid_path dataset\\bitcoin_sent_valid.csv "
                    "--models modernbert cryptobert finbert bert-base roberta-base",
                ]
            )
        )

        st.markdown("**Objective**")
        st.write(
            "This platform helps quant researchers compare NLP models on a fixed dataset, "
            "then select the best model by balancing accuracy and latency trade‑offs."
        )

    with tab_overview:
        st.subheader("Quick Metrics (from benchmark_results.csv)")
        benchmark = load_csv(BASE_DIR / "benchmark_results.csv")
        if benchmark.empty:
            st.warning("benchmark_results.csv is empty. Run experiments/benchmark.py first.")
        else:
            ranked = benchmark.sort_values("f1_macro", ascending=False).reset_index(drop=True)
            top = ranked.iloc[0]
            best_acc = benchmark.loc[benchmark["accuracy"].idxmax()]
            fastest = benchmark.loc[benchmark["latency_ms_per_tweet"].idxmin()]
            cols = st.columns(4)
            cols[0].metric("Models", f"{benchmark['model'].nunique()}")
            cols[1].metric("Top Macro F1", f"{top['f1_macro']:.4f}")
            cols[2].metric("Best Accuracy", f"{best_acc['accuracy']:.4f}")
            cols[3].metric("Fastest (ms)", f"{fastest['latency_ms_per_tweet']:.2f}")

            st.subheader("Model Leaderboard (Macro F1)")
            st.dataframe(format_table(ranked), use_container_width=True)

    with tab_benchmark:
        st.subheader("Benchmark Results")
        benchmark = load_csv(BASE_DIR / "benchmark_results.csv")
        if benchmark.empty:
            st.warning("benchmark_results.csv is empty. Run experiments/benchmark.py first.")
        else:
            st.caption(f"Models found: {benchmark['model'].nunique()}")
            st.markdown("**Model snapshot**")
            model_choice = st.selectbox("Choose model", benchmark["model"].tolist())
            row = benchmark[benchmark["model"] == model_choice].iloc[0]
            cols = st.columns(4)
            cols[0].metric("Accuracy", f"{row['accuracy']:.4f}")
            cols[1].metric("Macro F1", f"{row['f1_macro']:.4f}")
            cols[2].metric("Latency (ms)", f"{row['latency_ms_per_tweet']:.2f}")
            cols[3].metric("Models", f"{benchmark['model'].nunique()}")

            metric_options = [
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "latency_ms_per_tweet",
            ]
            metric = st.selectbox("Metric", metric_options, index=3)
            chart = chart_bar(benchmark, "model", metric, f"{metric} by model")
            st.altair_chart(chart, use_container_width=True)

            if {"latency_ms_per_tweet", "f1_macro"}.issubset(benchmark.columns):
                st.subheader("Accuracy vs Latency")
                scatter = chart_scatter(
                    benchmark,
                    "latency_ms_per_tweet",
                    "f1_macro",
                    "Latency vs Macro F1",
                )
                st.altair_chart(scatter, use_container_width=True)

            if "slang_accuracy" in benchmark.columns:
                benchmark = benchmark.drop(columns=["slang_accuracy"])
            st.dataframe(format_table(benchmark), use_container_width=True)

            st.subheader("Saved Plots")
            plot_cols = st.columns(2)
            acc_plot = RESULTS_DIR / "accuracy_vs_latency.png"
            f1_plot = RESULTS_DIR / "f1_vs_latency.png"
            if acc_plot.exists():
                plot_cols[0].image(str(acc_plot), caption="Accuracy vs Latency", use_container_width=True)
            else:
                plot_cols[0].info("accuracy_vs_latency.png not found.")
            if f1_plot.exists():
                plot_cols[1].image(str(f1_plot), caption="Macro F1 vs Latency", use_container_width=True)
            else:
                plot_cols[1].info("f1_vs_latency.png not found.")

    with tab_error:
        st.subheader("Error Breakdown")
        confusion = load_csv(RESULTS_DIR / "confusion_pairs.csv")
        if not confusion.empty:
            if "is_error" in confusion.columns:
                confusion = confusion[confusion["is_error"] == 1]
            st.markdown("**Confusion pairs**")
            st.dataframe(
                confusion[
                    ["true_label_str", "pred_label_str", "count"]
                ].reset_index(drop=True),
                use_container_width=True,
            )

        length_df = load_csv(RESULTS_DIR / "error_by_length.csv")
        if not length_df.empty:
            st.markdown("**Error rate by length**")
            chart = chart_bar(length_df, "length_bucket", "error_rate", "Error rate by length")
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(format_table(length_df), use_container_width=True)

        signal_df = load_csv(RESULTS_DIR / "error_by_signal.csv")
        if not signal_df.empty:
            st.markdown("**Error rate by signal**")
            signal_df = signal_df.copy()
            signal_df["value"] = signal_df["value"].map({0: "no", 1: "yes", "0": "no", "1": "yes"})
            chart = chart_bar(signal_df, "signal", "error_rate", "Error rate by signal")
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(format_table(signal_df), use_container_width=True)

        conf_df = load_csv(RESULTS_DIR / "error_by_confidence.csv")
        if not conf_df.empty:
            st.markdown("**Error rate by confidence bucket**")
            chart = chart_bar(conf_df, "confidence_bucket", "error_rate", "Error rate by confidence")
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(format_table(conf_df), use_container_width=True)

        st.subheader("Model Comparison (Accuracy vs F1)")
        benchmark = load_csv(BASE_DIR / "benchmark_results.csv")
        if benchmark.empty:
            st.info("benchmark_results.csv is empty. Run experiments/benchmark.py first.")
        else:
            compare_cols = ["accuracy", "f1_macro", "f1_weighted"]
            compare_df = benchmark[["model"] + compare_cols].melt(
                id_vars="model",
                value_vars=compare_cols,
                var_name="metric",
                value_name="score",
            )
            chart = (
                alt.Chart(compare_df, title="Accuracy and F1 across models")
                .mark_bar()
                .encode(
                    x=alt.X("model:N", sort=None),
                    y=alt.Y("score:Q"),
                    color=alt.Color("metric:N"),
                    tooltip=["model", "metric", "score"],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    with tab_artifacts:
        st.subheader("Key Metrics Dashboard")
        benchmark = load_csv(BASE_DIR / "benchmark_results.csv")
        if benchmark.empty:
            st.info("benchmark_results.csv is empty. Run experiments/benchmark.py first.")
        else:
            cols = st.columns(3)
            best_f1 = benchmark.loc[benchmark["f1_macro"].idxmax()]
            best_acc = benchmark.loc[benchmark["accuracy"].idxmax()]
            fastest = benchmark.loc[benchmark["latency_ms_per_tweet"].idxmin()]
            cols[0].metric("Top Macro F1", f"{best_f1['model']} ({best_f1['f1_macro']:.4f})")
            cols[1].metric("Best Accuracy", f"{best_acc['model']} ({best_acc['accuracy']:.4f})")
            cols[2].metric("Fastest", f"{fastest['model']} ({fastest['latency_ms_per_tweet']:.2f} ms)")

            metric_cols = st.columns(2)
            chart1 = chart_bar(benchmark, "model", "accuracy", "Accuracy by model")
            chart2 = chart_bar(benchmark, "model", "f1_macro", "Macro F1 by model")
            metric_cols[0].altair_chart(chart1, use_container_width=True)
            metric_cols[1].altair_chart(chart2, use_container_width=True)

    with tab_misclass:
        st.subheader("Misclassified Samples")
        index_df = load_csv(RESULTS_DIR / "misclassified_pairs" / "index.csv")
        if not index_df.empty:
            index_df = index_df.sort_values("count", ascending=False)
            options = {
                f"{row.true_label_str} -> {row.pred_label_str} ({row['count']})": row["csv_path"]
                for _, row in index_df.iterrows()
            }
            choice = st.selectbox("Confusion pair", list(options.keys()))
            pair_path = RESULTS_DIR / options[choice]
            rows = load_csv(pair_path)
        else:
            rows = load_csv(RESULTS_DIR / "misclassified_samples.csv")

        if rows.empty:
            st.info("No misclassification samples found.")
        else:
            rows = rows[
                ["text", "true_label_str", "pred_label_str", "pred_confidence", "topk_probs"]
            ].copy()
            rows["pred_confidence"] = rows["pred_confidence"].round(4)
            st.dataframe(rows, use_container_width=True, height=600)


if __name__ == "__main__":
    main()
