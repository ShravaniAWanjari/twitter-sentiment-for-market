
import os
import sys
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# Add research root to sys.path
sys.path.append(os.path.abspath('.'))

from research.model_factory import load_model
from research.core.preprocessing import align_label

def get_cm_data():
    valid_path = 'research/data/sent_valid.csv'
    df = pd.read_csv(valid_path)
    labels = [align_label(lbl) for lbl in df['label']]
    class_names = ['Bearish', 'Neutral', 'Bullish']
    
    models_to_test = ['modernbert', 'cryptobert', 'finbert']
    results = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name in models_to_test:
        try:
            run_path = f"experiments/runs/{model_name}/final_model"
            if not os.path.exists(run_path) and model_name == 'modernbert':
                run_path = "experiments/modernbert_runs/final_model"
            
            if os.path.exists(run_path):
                model_wrapper = load_model(model_name, model_name=run_path, device=device)
            else:
                model_wrapper = load_model(model_name, device=device)
            
            model = model_wrapper.model
            tokenizer = model_wrapper.tokenizer
            model.eval()
            
            all_preds = []
            with torch.no_grad():
                for start in range(0, len(df), 32):
                    batch = df.iloc[start : start + 32]
                    texts = batch['text'].astype(str).tolist()
                    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
                    outputs = model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                    all_preds.extend(preds)
            
            cm = confusion_matrix(labels, all_preds)
            results[model_name] = cm
        except Exception as e:
            print(f"Error {model_name}: {e}")
            
    # Format and saved to a temp file or print to read
    with open('research/results/cm_tables.txt', 'w') as f:
        for model_name, cm in results.items():
            f.write(f"\n### {model_name} Confusion Matrix\n")
            f.write("| True \\ Pred | Bearish | Neutral | Bullish |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            f.write(f"| **Bearish** | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |\n")
            f.write(f"| **Neutral** | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |\n")
            f.write(f"| **Bullish** | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} |\n")

if __name__ == "__main__":
    get_cm_data()
