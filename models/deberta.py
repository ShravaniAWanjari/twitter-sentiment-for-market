from __future__ import annotations

from typing import Any, Optional

from core import CryptoTransformerBase, LABEL_ID2STR, LABEL_STR2ID


class DeBERTaV3Wrapper(CryptoTransformerBase):
    """DeBERTa-v3 classifier wrapper (smaller variant for stability)."""

    DEFAULT_MODEL_ID = "microsoft/deberta-v3-small"

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        device=None,
        max_length: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name or self.DEFAULT_MODEL_ID,
            device=device,
            max_length=max_length,
            id2label=LABEL_ID2STR,
            label2id=LABEL_STR2ID,
            **kwargs,
        )
