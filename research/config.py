from transformers import AutoConfig
from core.preprocessing import LABEL_ID2STR, LABEL_STR2ID

model_dir = "experiments/modernbert_runs/final_model"
config = AutoConfig.from_pretrained(
    "answerdotai/ModernBERT-base",
    num_labels=3,
    id2label=LABEL_ID2STR,
    label2id=LABEL_STR2ID,
    trust_remote_code=True,
)
config.save_pretrained(model_dir)
print("Saved config.json to", model_dir)