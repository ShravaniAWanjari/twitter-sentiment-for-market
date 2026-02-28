
import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Add research root to sys.path
sys.path.append(os.path.abspath('.'))
from research.core.preprocessing import preprocess_text, align_label

def calculate_ece(logits, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('research/data/sent_valid.csv')
    labels = torch.tensor([align_label(lbl) for lbl in df['label']]).to(device)
    
    models = {
        'ModernBERT': 'experiments/modernbert_runs/final_model',
        'CryptoBERT': 'experiments/runs/cryptobert/final_model'
    }
    
    results = []
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path): continue
        print(f"Evaluating {model_name} for Calibration & Imbalance...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        
        all_logits = []
        all_preds = []
        
        with torch.no_grad():
            for start in range(0, len(df), 32):
                batch = df.iloc[start : start + 32]
                processed_texts = [preprocess_text(t) for t in batch['text'].astype(str)]
                inputs = tokenizer(processed_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
                outputs = model(**inputs)
                all_logits.append(outputs.logits)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.append(preds)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        
        ece = calculate_ece(all_logits, labels)
        f1 = f1_score(labels.cpu(), all_preds.cpu(), average='macro', zero_division=0)
        acc = accuracy_score(labels.cpu(), all_preds.cpu())
        
        # Per-class F1
        per_class_f1 = f1_score(labels.cpu(), all_preds.cpu(), average=None, zero_division=0)
        
        results.append({
            'Model': model_name,
            'ECE': ece,
            'Macro F1': f1,
            'Acc': acc,
            'F1_Bearish': per_class_f1[0],
            'F1_Neutral': per_class_f1[1],
            'F1_Bullish': per_class_f1[2] if len(per_class_f1) > 2 else 0.0
        })

    res_df = pd.DataFrame(results)
    os.makedirs('research/results/calibration', exist_ok=True)
    res_df.to_csv('research/results/calibration/calibration_results.csv', index=False)
    
    with open('research/results/calibration/calibration_report.md', 'w') as f:
        f.write("# Calibration & Class-Imbalance Report\n\n")
        f.write("Expected Calibration Error (ECE) determines how well the model's confidence reflects its accuracy.\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## Analysis\n")
        f.write("- **ECE**: Lower is better. High ECE suggests overconfidence or underconfidence.\n")
        f.write("- **Per-class F1**: Helps identify if the model is biased towards the majority class (usually Neutral).\n")

if __name__ == "__main__":
    run_experiment()
