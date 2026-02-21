from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.metrics import compute_classification_metrics
from core.preprocessing import align_label, tokenize_with_preprocessing
from model_factory import load_model


@dataclass
class TrainingConfig:
    model: str
    train_path: Path
    valid_path: Path
    output_dir: Path
    epochs: int = 2
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_length: int = 256
    use_flash_attention: bool = False


MODEL_TRAINING_PROFILES = {
    "modernbert": {"learning_rate": 2e-5, "max_length": 256, "fp16": True},
    "cryptobert": {"learning_rate": 2e-5, "max_length": 256, "fp16": True},
    "finbert": {"learning_rate": 2e-5, "max_length": 256, "fp16": True},
    "deberta-v3": {"learning_rate": 5e-6, "max_length": 128, "fp16": True},
    "deberta": {"learning_rate": 5e-6, "max_length": 128, "fp16": True},
    "bert-base": {"learning_rate": 2e-5, "max_length": 256, "fp16": True},
    "bert": {"learning_rate": 2e-5, "max_length": 256, "fp16": True},
    "roberta-base": {"learning_rate": 2e-5, "max_length": 256, "fp16": True},
    "roberta": {"learning_rate": 2e-5, "max_length": 256, "fp16": True},
}


class TokenizedTweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        texts: List[str] = df["text"].astype(str).tolist()
        labels: List[int] = [align_label(lbl) for lbl in df["label"]]
        encodings = tokenize_with_preprocessing(
            tokenizer,
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class SafeTrainer(Trainer):
    """Trainer that sanitizes non-finite gradients to keep training stable."""

    def __init__(self, *args, max_bad_steps: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self._bad_steps = 0
        self._max_bad_steps = max_bad_steps

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if not torch.isfinite(loss):
            self._bad_steps += 1
            if self._bad_steps >= self._max_bad_steps:
                raise RuntimeError("Non-finite loss encountered repeatedly; aborting training.")
            return loss.detach() * 0
        self.accelerator.backward(loss)
        for param in model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                param.grad.data.nan_to_num_(0.0, posinf=0.0, neginf=0.0)
        return loss.detach()


def build_datasets(config: TrainingConfig, tokenizer, max_length: int):
    train_df = pd.read_csv(config.train_path)
    valid_df = pd.read_csv(config.valid_path)
    train_ds = TokenizedTweetDataset(train_df, tokenizer, max_length)
    valid_ds = TokenizedTweetDataset(valid_df, tokenizer, max_length)
    return train_ds, valid_ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train any registered transformer model.")
    parser.add_argument("--model", type=str, default="modernbert")
    parser.add_argument("--train_path", type=Path, default=Path("dataset/bitcoin_sent_train.csv"))
    parser.add_argument("--valid_path", type=Path, default=Path("dataset/bitcoin_sent_valid.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use-flash", action="store_true", help="Enable Flash Attention 2.0 if available.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 even if CUDA is available.")
    parser.add_argument("--sanity-check", action="store_true", help="Run a single-batch loss check and exit.")
    parser.add_argument("--debug-step", action="store_true", help="Run a single forward/backward step and report NaNs.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--preflight", action="store_true", help="Run a safety preflight and auto-adjust settings.")
    parser.add_argument("--max_bad_steps", type=int, default=5)
    args = parser.parse_args()

    model_key = args.model.lower()
    profile = MODEL_TRAINING_PROFILES.get(model_key, {})

    run_dir = args.output_dir / args.model
    config = TrainingConfig(
        model=args.model,
        train_path=args.train_path,
        valid_path=args.valid_path,
        output_dir=run_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=profile.get("learning_rate", args.learning_rate),
        weight_decay=args.weight_decay,
        max_length=profile.get("max_length", args.max_length),
        use_flash_attention=args.use_flash,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_flash = config.use_flash_attention and torch.cuda.is_available()

    model = load_model(
        config.model,
        use_flash_attention=use_flash,
        device=device,
        max_length=config.max_length,
    )

    if use_flash:
        try:
            torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
        except Exception:
            pass

    if args.sanity_check or args.debug_step:
        train_df = pd.read_csv(config.train_path)
        train_ds = TokenizedTweetDataset(train_df, model.tokenizer, config.max_length)
        loader = DataLoader(train_ds, batch_size=min(8, len(train_ds)))
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        if args.debug_step:
            model.train()
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Debug step failed: loss is not finite.")
            loss.backward()
            bad_param = None
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    bad_param = name
                    break
            if bad_param:
                raise RuntimeError(f"Debug step failed: non-finite gradients in {bad_param}.")
            print(f"Debug step OK: loss={loss.item():.4f}")
            return
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        if not torch.isfinite(loss):
            raise RuntimeError("Sanity check failed: loss is not finite.")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Sanity check failed: logits contain NaNs or infs.")
        print(f"Sanity check OK: loss={loss.item():.4f}, logits range=({logits.min().item():.3f}, {logits.max().item():.3f})")
        return

    train_ds, valid_ds = build_datasets(config, model.tokenizer, config.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer)

    import inspect

    warmup_steps = args.warmup_steps
    if warmup_steps <= 0 and args.warmup_ratio > 0:
        steps_per_epoch = max(1, (len(train_ds) + config.train_batch_size - 1) // config.train_batch_size)
        warmup_steps = int(steps_per_epoch * config.epochs * args.warmup_ratio)

    requested_fp16 = torch.cuda.is_available() and not args.no_fp16
    if args.no_fp16:
        use_fp16 = False
    else:
        use_fp16 = bool(profile.get("fp16", requested_fp16)) if args.preflight else requested_fp16
    arg_options = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1_macro",
        "greater_is_better": True,
        "fp16": use_fp16,
        "max_grad_norm": args.max_grad_norm,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "remove_unused_columns": False,
        "report_to": "none",
    }
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered_args = {k: v for k, v in arg_options.items() if k in supported}
    if "evaluation_strategy" not in filtered_args:
        for key in ("save_strategy", "load_best_model_at_end", "metric_for_best_model", "greater_is_better"):
            filtered_args.pop(key, None)
    training_args = TrainingArguments(**filtered_args)

    def run_preflight():
        train_df = pd.read_csv(config.train_path)
        train_ds = TokenizedTweetDataset(train_df, model.tokenizer, config.max_length)
        loader = DataLoader(train_ds, batch_size=min(8, len(train_ds)))
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        model.train()
        outputs = model(**batch)
        loss = outputs.loss
        if not torch.isfinite(loss):
            return False, "loss is not finite"
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False, f"non-finite gradients in {name}"
        return True, "ok"

    if args.preflight:
        ok, message = run_preflight()
        if not ok and use_fp16:
            use_fp16 = False
            filtered_args["fp16"] = False
            training_args = TrainingArguments(**filtered_args)
            ok, message = run_preflight()
        if not ok:
            config.learning_rate = min(config.learning_rate, 3e-6)
            filtered_args["learning_rate"] = config.learning_rate
            training_args = TrainingArguments(**filtered_args)
            ok, message = run_preflight()
        if not ok:
            raise RuntimeError(f"Preflight failed: {message}. Try lowering batch size or max_length.")

    def compute_metrics_fn(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        preds = logits.argmax(axis=-1)
        metrics = compute_classification_metrics(labels, preds)
        return {f"eval_{k}": v for k, v in metrics.items()}

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        max_bad_steps=args.max_bad_steps,
    )

    trainer.train()
    trainer.save_model(str(config.output_dir / "final_model"))
    if hasattr(model.model, "config") and model.model.config is not None:
        model.model.config.save_pretrained(str(config.output_dir / "final_model"))
    model.tokenizer.save_pretrained(str(config.output_dir / "final_model"))


if __name__ == "__main__":
    main()
