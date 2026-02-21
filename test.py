import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report, confusion_matrix

model_dir = "experiments/modernbert_runs/final_model"
data_path = "dataset/bitcoin_sent_valid.csv"

df = pd.read_csv(data_path)
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

def batch_iter(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

preds = []
with torch.no_grad():
    for batch in batch_iter(texts, 32):
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

print(classification_report(labels, preds, digits=4))
print("Confusion matrix:\n", confusion_matrix(labels, preds))
