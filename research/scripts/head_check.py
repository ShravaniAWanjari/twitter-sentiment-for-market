
import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Root path for imports
sys.path.append(os.path.abspath('.'))
from research.core.preprocessing import align_label

def evaluate_condition(model_id, weights_path=None, forced_reinit=False, device='cpu'):
    print(f"\nEvaluating {model_id} | Path: {weights_path} | Reinit: {forced_reinit}")
    
    if forced_reinit:
        # Load with 3 labels to force a mismatch/reinit if the original had different
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3, ignore_mismatched_sizes=True)
    elif weights_path:
        model = AutoModelForSequenceClassification.from_pretrained(weights_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)
    model.eval()
    
    df = pd.read_csv('research/data/sent_valid.csv')
    labels = [align_label(lbl) for lbl in df['label']]
    
    all_preds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), 32)):
            batch = df.iloc[start : start + 32]
            inputs = tokenizer(batch['text'].astype(str).tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            
    acc = accuracy_score(labels, all_preds)
    f1 = f1_score(labels, all_preds, average='macro', zero_division=0)
    return acc, f1

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    domain_models = {
        'CryptoBERT': 'ElKulako/cryptobert',
        'FinBERT': 'ProsusAI/finbert'
    }
    
    for name, model_id in domain_models.items():
        # 1. As-is
        try:
            acc, f1 = evaluate_condition(model_id, device=device)
            results.append({'Model': name, 'Setting': 'As-is Pretrained', 'Acc': acc, 'Macro F1': f1})
        except Exception as e:
            print(f"Error as-is {name}: {e}")

        # 2. Forced Reinit
        try:
            acc, f1 = evaluate_condition(model_id, forced_reinit=True, device=device)
            results.append({'Model': name, 'Setting': 'Forced Reinit', 'Acc': acc, 'Macro F1': f1})
        except Exception as e:
            print(f"Error reinit {name}: {e}")
            
        # 3. Fine-tuned (from experiments/runs)
        run_path = f"experiments/runs/{name.lower()}/final_model"
        if os.path.exists(run_path):
            try:
                acc, f1 = evaluate_condition(model_id, weights_path=run_path, device=device)
                results.append({'Model': name, 'Setting': 'Fine-tuned', 'Acc': acc, 'Macro F1': f1})
            except Exception as e:
                print(f"Error fine-tuned {name}: {e}")
        else:
            results.append({'Model': name, 'Setting': 'Fine-tuned', 'Acc': 'N/A', 'Macro F1': 'N/A'})

    res_df = pd.DataFrame(results)
    os.makedirs('research/results/head_check', exist_ok=True)
    res_df.to_csv('research/results/head_check/head_results.csv', index=False)
    print("\nHead Preservation Results:")
    print(res_df.to_string())
    
    with open('research/results/head_check/head_report.md', 'w') as f:
        f.write("# Head Preservation Experiment Report\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## Analysis\n")
        f.write("If 'As-is Pretrained' performance is significantly higher than 'Fine-tuned', it suggests the fine-tuning pipeline inadvertently learned a corrupted mapping or reinitialized the head poorly.\n")

if __name__ == "__main__":
    run_experiment()
