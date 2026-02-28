from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int]
) -> Dict[str, float]:
    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
