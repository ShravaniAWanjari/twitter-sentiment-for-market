from __future__ import annotations

from typing import Dict, Type

from models import (
    BertBaseWrapper,
    CryptoBERTWrapper,
    DeBERTaV3Wrapper,
    FinBERTWrapper,
    ModernBERTWrapper,
    RoBERTaBaseWrapper,
)

MODEL_REGISTRY: Dict[str, Type] = {
    "modernbert": ModernBERTWrapper,
    "cryptobert": CryptoBERTWrapper,
    "finbert": FinBERTWrapper,
    "deberta-v3": DeBERTaV3Wrapper,
    "deberta": DeBERTaV3Wrapper,
    "bert-base": BertBaseWrapper,
    "bert": BertBaseWrapper,
    "roberta-base": RoBERTaBaseWrapper,
    "roberta": RoBERTaBaseWrapper,
}

DEFAULT_MODEL_IDS: Dict[str, str] = {
    "modernbert": ModernBERTWrapper.DEFAULT_MODEL_ID,
    "cryptobert": CryptoBERTWrapper.DEFAULT_MODEL_ID,
    "finbert": FinBERTWrapper.DEFAULT_MODEL_ID,
    "deberta-v3": DeBERTaV3Wrapper.DEFAULT_MODEL_ID,
    "bert-base": BertBaseWrapper.DEFAULT_MODEL_ID,
    "bert": BertBaseWrapper.DEFAULT_MODEL_ID,
    "roberta-base": RoBERTaBaseWrapper.DEFAULT_MODEL_ID,
    "roberta": RoBERTaBaseWrapper.DEFAULT_MODEL_ID,
}


def load_model(name: str, **kwargs):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered. Available: {list(MODEL_REGISTRY)}")
    if "model_name" not in kwargs and key in DEFAULT_MODEL_IDS:
        kwargs["model_name"] = DEFAULT_MODEL_IDS[key]
    wrapper_cls = MODEL_REGISTRY[key]
    return wrapper_cls(**kwargs)


def available_models():
    return list(MODEL_REGISTRY.keys())
