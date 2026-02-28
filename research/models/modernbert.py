from __future__ import annotations

from typing import Any, Optional

from core import CryptoTransformerBase, LABEL_ID2STR, LABEL_STR2ID


class ModernBERTWrapper(CryptoTransformerBase):
    """ModernBERT sequence classifier with optional Flash Attention 2.0."""

    DEFAULT_MODEL_ID = "answerdotai/ModernBERT-base"

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        use_flash_attention: bool = True,
        torch_dtype=None,
        device=None,
        max_length: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name or self.DEFAULT_MODEL_ID,
            use_flash_attention=use_flash_attention,
            torch_dtype=torch_dtype,
            device=device,
            max_length=max_length,
            id2label=LABEL_ID2STR,
            label2id=LABEL_STR2ID,
            **kwargs,
        )
