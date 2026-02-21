from transformers import pipeline

model_dir = "experiments/modernbert_runs/final_model"
clf = pipeline("text-classification", model=model_dir, tokenizer=model_dir, top_k=None)

samples = [
    "Bitcoin rallies after ETF approval rumors.",
    "Market crashes as exchange halts withdrawals.",
    "Price stable despite low volume."
]

for s in samples:
    print(s, "->", clf(s))