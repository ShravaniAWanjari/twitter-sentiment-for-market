from __future__ import annotations

import html
import re
from typing import Iterable, List, MutableMapping, Optional

LABEL_ID2STR = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
LABEL_STR2ID = {v.lower(): k for k, v in LABEL_ID2STR.items()}

_LABEL_ALIASES = {
    "bear": LABEL_STR2ID["bearish"],
    "bearish": LABEL_STR2ID["bearish"],
    "negative": LABEL_STR2ID["bearish"],
    "sell": LABEL_STR2ID["bearish"],
    "0": LABEL_STR2ID["bearish"],
    0: LABEL_STR2ID["bearish"],
    -1: LABEL_STR2ID["bearish"],
    "neutral": LABEL_STR2ID["neutral"],
    "1": LABEL_STR2ID["neutral"],
    1: LABEL_STR2ID["neutral"],
    "bull": LABEL_STR2ID["bullish"],
    "bullish": LABEL_STR2ID["bullish"],
    "positive": LABEL_STR2ID["bullish"],
    "buy": LABEL_STR2ID["bullish"],
    "2": LABEL_STR2ID["bullish"],
    2: LABEL_STR2ID["bullish"],
}

_URL_PATTERN = re.compile(r"http[s]?://\\S+")
_MULTISPACE_PATTERN = re.compile(r"\\s+")
_HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]+")

_SLANG_TABLE: MutableMapping[str, str] = {
    "hodl": "hold",
    "rekt": "wrecked",
    "moon": "moon",
    "mooning": "surging",
    "pump": "pump",
    "dump": "dump",
    "bulls": "bulls",
    "bears": "bears",
    "ath": "all time high",
    "fomo": "fear of missing out",
    "fud": "fear uncertainty doubt",
}


def preprocess_text(text: str) -> str:
    """Lightweight tweet cleaning shared by all models."""
    text = html.unescape(text or "")
    text = _URL_PATTERN.sub("", text)
    text = _HANDLE_PATTERN.sub("@user", text)
    lowered = text.lower()
    for slang, normalized in _SLANG_TABLE.items():
        lowered = lowered.replace(slang, normalized)
    cleaned = _MULTISPACE_PATTERN.sub(" ", lowered).strip()
    return cleaned


def align_label(raw_label) -> int:
    """Map dataset-specific labels onto the unified Bearish/Neutral/Bullish schema."""
    key = raw_label
    if isinstance(raw_label, str):
        key = raw_label.strip().lower()
    if key not in _LABEL_ALIASES:
        raise ValueError(f"Unrecognized label '{raw_label}'")
    return _LABEL_ALIASES[key]


def tokenize_with_preprocessing(
    tokenizer,
    texts: Iterable[str],
    *,
    max_length: int = 256,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "pt",
    **kwargs,
):
    cleaned: List[str] = [preprocess_text(t) for t in texts]
    return tokenizer(
        cleaned,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
        **kwargs,
    )
