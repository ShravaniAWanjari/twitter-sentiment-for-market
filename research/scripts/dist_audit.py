
import os
import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Add research root to sys.path
sys.path.append(os.path.abspath('.'))

from research.model_factory import load_model, MODEL_REGISTRY
from research.core.preprocessing import align_label

def run_audit():
    # Setup paths
    valid_path = 'research/data/sent_valid.csv'
    results_dir = 'research/results/dist_audit'
    os.makedirs(results_dir, exist_ok=True)
    
    df = pd.read_csv(valid_path)
    labels = [align_label(lbl) for lbl in df['label']]
    class_names = ['Bearish', 'Neutral', 'Bullish']
    
    audit_summary = []
    
    models_to_test = ['modernbert', 'cryptobert', 'finbert', 'bert-base', 'roberta-base']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    for model_name in models_to_test:
        print(f"Auditing {model_name}...")
        try:
            # If a fine-tuned model exists in experiments/runs, load that.
            run_path = f"experiments/runs/{model_name}/final_model"
            if not os.path.exists(run_path):
                if model_name == 'modernbert':
                    run_path = "experiments/modernbert_runs/final_model"
            
            if os.path.exists(run_path):
                print(f"Loading fine-tuned weights from {run_path}")
                model_wrapper = load_model(model_name, model_name=run_path, device=device)
            else:
                print(f"Warning: Fine-tuned weights not found for {model_name}. Loading base.")
                model_wrapper = load_model(model_name, device=device)
            
            model = model_wrapper.model
            tokenizer = model_wrapper.tokenizer
            model.eval()
            
            all_preds = []
            
            with torch.no_grad():
                for start in tqdm(range(0, len(df), 32)):
                    batch = df.iloc[start : start + 32]
                    texts = batch['text'].astype(str).tolist()
                    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
                    outputs = model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                    all_preds.extend(preds)
            
            # 1. Prediction Statistics
            pred_counts = pd.Series(all_preds).value_counts(normalize=True).sort_index()
            # Ensure all classes are present in index
            for i in range(3):
                if i not in pred_counts: pred_counts[i] = 0.0
            pred_counts = pred_counts.sort_index()
            
            is_majority = "Yes" if (pred_counts > 0.9).any() else "No"
            
            # 2. Metrics
            report = classification_report(labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
            
            audit_summary.append({
                'Model': model_name,
                '%Bear': f"{pred_counts[0]*100:.1f}%",
                '%Neut': f"{pred_counts[1]*100:.1f}%",
                '%Bull': f"{pred_counts[2]*100:.1f}%",
                'Majority?': is_majority,
                'Acc': report['accuracy'],
                'Macro F1': report['macro avg']['f1-score'],
                'Macro Recall': report['macro avg']['recall']
            })
            
            # 3. Confusion Matrix Plot
            cm = confusion_matrix(labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
            plt.title(f"Confusion Matrix: {model_name}")
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.savefig(f"{results_dir}/{model_name}_cm.png")
            plt.close()

        except Exception as e:
            print(f"Error auditing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save overall summary
    summary_df = pd.DataFrame(audit_summary)
    summary_df.to_csv(f"{results_dir}/audit_table.csv", index=False)
    
    with open(f"{results_dir}/audit_report.md", "w") as f:
        f.write("# Prediction Distribution Audit Report\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Observations\n")
        for model in audit_summary:
            if model['Macro Recall'] < 0.35:
                f.write(f"- **{model['Model']}**: Suspiciously low macro recall ({model['Macro Recall']:.3f}). Check for class collapse or label mapping issues.\n")
            if model['Majority?'] == 'Yes':
                f.write(f"- **{model['Model']}**: Predominantly predicts one class. Likely a training or head reinit issue.\n")

if __name__ == "__main__":
    run_audit()
