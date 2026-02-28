
import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score
from tqdm import tqdm
import html
import re

# Root path for imports
sys.path.append(os.path.abspath('.'))
from research.core.preprocessing import align_label

_URL_PATTERN = re.compile(r"http[s]?://\\S+")
_MULTISPACE_PATTERN = re.compile(r"\\s+")
_HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
_SLANG_TABLE = {
    "hodl": "hold", "rekt": "wrecked", "moon": "moon", "mooning": "surging",
    "pump": "pump", "dump": "dump", "bulls": "bulls", "bears": "bears",
    "ath": "all time high", "fomo": "fear of missing out", "fud": "fear uncertainty doubt",
}

def preprocess_custom(text, slang=True, url=True, handle=True):
    text = html.unescape(text or "")
    if url: text = _URL_PATTERN.sub("", text)
    if handle: text = _HANDLE_PATTERN.sub("@user", text)
    lowered = text.lower()
    if slang:
        for s, n in _SLANG_TABLE.items():
            lowered = lowered.replace(s, n)
    return _MULTISPACE_PATTERN.sub(" ", lowered).strip()

def evaluate_ablation(model, tokenizer, df, slang=True, url=True, handle=True, device='cpu'):
    labels = [align_label(lbl) for lbl in df['label']]
    all_preds = []
    
    with torch.no_grad():
        for start in range(0, len(df), 32):
            batch = df.iloc[start : start + 32]
            processed_texts = [preprocess_custom(t, slang, url, handle) for t in batch['text'].astype(str)]
            inputs = tokenizer(processed_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            
    return f1_score(labels, all_preds, average='macro', zero_division=0)

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    models = {
        'ModernBERT': 'experiments/modernbert_runs/final_model',
        'CryptoBERT': 'experiments/runs/cryptobert/final_model'
    }
    
    # Files to test on
    test_files = {
        'Main Valid': 'research/data/sent_valid.csv',
        'Slang Set': 'research/data/sent_slang.csv'
    }
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}, path {model_path} not found")
            continue
            
        print(f"\nAblating {model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        
        for file_label, file_path in test_files.items():
            if not os.path.exists(file_path): continue
            df = pd.read_csv(file_path)
            
            # Ablations
            conditions = [
                ('Full Preprocessing', True, True, True),
                ('No Slang Normalization', False, True, True),
                ('No URL Removal', True, False, True),
                ('No Handle Normalization', True, True, False),
                ('Raw Text (No Pre)', False, False, False)
            ]
            
            for cond_name, s, u, h in conditions:
                f1 = evaluate_ablation(model, tokenizer, df, slang=s, url=u, handle=h, device=device)
                results.append({
                    'Model': model_name,
                    'Data': file_label,
                    'Ablation': cond_name,
                    'Macro F1': f1
                })
                print(f"{model_name} | {file_label} | {cond_name}: {f1:.4f}")

    res_df = pd.DataFrame(results)
    os.makedirs('research/results/ablation', exist_ok=True)
    res_df.to_csv('research/results/ablation/ablation_results.csv', index=False)
    
    with open('research/results/ablation/ablation_report.md', 'w') as f:
        f.write("# Preprocessing Ablation Report\n\n")
        f.write(res_df.to_markdown(index=False))

if __name__ == "__main__":
    run_experiment()
