from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .preprocessing import preprocess_text


class CryptoTransformerBase(nn.Module):
    """
    Abstract wrapper that standardizes model access for benchmarking and XAI.

    - Exposes embedding and attention modules for attribution hooks.
    - Forces output_hidden_states/output_attentions on every forward pass.
    - Supports forwarding from differentiable input embeddings (Captum friendly).
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        *,
        num_labels: int = 3,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        use_flash_attention: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 256,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        tokenizer_target = tokenizer_name or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_target, use_fast=True)

        config_kwargs: Dict[str, Any] = {
            "output_hidden_states": True,
            "output_attentions": True,
            "num_labels": num_labels,
        }
        if id2label:
            config_kwargs["id2label"] = id2label
        if label2id:
            config_kwargs["label2id"] = label2id

        config = None
        try:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **config_kwargs,
            )
        except KeyError:
            # Some community models (e.g., ModernBERT) rely on remote code; let the model loader infer config.
            pass

        load_kwargs: Dict[str, Any] = {**model_kwargs}
        if torch_dtype:
            load_kwargs["torch_dtype"] = torch_dtype
        if use_flash_attention:
            if not torch.cuda.is_available():
                use_flash_attention = False
            else:
                flash_attn_available = False
                try:
                    from transformers.utils import is_flash_attn_2_available
                except Exception:
                    is_flash_attn_2_available = None
                if is_flash_attn_2_available is not None:
                    try:
                        flash_attn_available = bool(is_flash_attn_2_available())
                    except Exception:
                        flash_attn_available = False
                else:
                    try:
                        import flash_attn  # noqa: F401
                    except Exception:
                        flash_attn_available = False
                    else:
                        flash_attn_available = True
                if not flash_attn_available:
                    use_flash_attention = False
        if use_flash_attention:
            # Hugging Face sets this through the attn_implementation kwarg.
            load_kwargs["attn_implementation"] = "flash_attention_2"
            if config is not None:
                config.attn_implementation = "flash_attention_2"
        else:
            if config is not None and getattr(config, "attn_implementation", None) is not None:
                config.attn_implementation = "eager"
            elif config is None:
                load_kwargs["attn_implementation"] = "eager"

        model_load_kwargs = dict(load_kwargs)
        if config is not None:
            model_load_kwargs["config"] = config

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            **model_load_kwargs,
        )

        if device:
            self.to(device)

    @property
    def embedding_layer(self) -> nn.Module:
        """Return the input embedding module for attribution hooks."""
        return self.model.get_input_embeddings()

    @property
    def attention_modules(self) -> List[nn.Module]:
        """
        Best-effort discovery of attention blocks for downstream introspection.
        Returns a list to allow consumers to register hooks per-layer/head.
        """
        base_model = getattr(self.model, "base_model", None)
        if base_model is None and hasattr(self.model, self.model.base_model_prefix):
            base_model = getattr(self.model, self.model.base_model_prefix)
        if base_model is None:
            base_model = self.model

        candidates = [
            ("encoder", "layer"),
            ("encoder", "layers"),
            ("transformer", "layer"),
            ("deberta", "encoder", "layer"),
        ]

        for path in candidates:
            module: Any = base_model
            for name in path:
                module = getattr(module, name, None)
                if module is None:
                    break
            if module is not None:
                try:
                    return list(module)
                except TypeError:
                    # Some implementations expose ModuleList-like objects without iteration support.
                    return [module]

        return []

    def encode_texts(
        self,
        texts: Iterable[str],
        *,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
        **tokenizer_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        cleaned = [preprocess_text(t) for t in texts]
        return self.tokenizer(
            cleaned,
            padding=padding,
            truncation=truncation,
            max_length=max_length or self.max_length,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )

    def embed_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Produce differentiable input embeddings for IG.
        The returned tensor has requires_grad=True and can be fed back via forward_from_embeds.
        """
        embeddings = self.embedding_layer(input_ids)
        return embeddings.detach().requires_grad_(True)

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        outputs = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            **kwargs,
        )

        # SequenceClassifierOutput is already Mapping-like, satisfying the dictionary contract
        # while remaining compatible with Hugging Face Trainer expectations.
        return outputs

    def forward_from_embeds(
        self,
        embeddings: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Captum-friendly forward using precomputed embeddings with gradients.
        """
        return self.forward(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
        )

    def to_device(self, device: Union[str, torch.device]) -> "CryptoTransformerBase":
        self.to(device)
        return self
