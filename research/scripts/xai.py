
import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients
import numpy as np

# Root path for imports
sys.path.append(os.path.abspath('.'))
from research.core.preprocessing import preprocess_text

# Samples for interpretability
XAI_SAMPLES = [
    "Bitcoin is mooning right now! HODL till we reach 100k!",
    "Just rekt by the latest dump. Bears are winning.",
    "The market is moving sideways with no clear direction.",
    "SEC announces new regulations for stablecoins.",
    "I'm not bullish on this pump. It feels like a trap.",
    "This is the best halving ever, diamond hands only."
]

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        # Captum needs just the tensor output
        outputs = self.model(input_ids)
        return outputs.logits

def get_attributions(text, model, tokenizer, device='cpu'):
    processed = preprocess_text(text)
    inputs = tokenizer(processed, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    input_ids = inputs['input_ids']
    
    wrapper = ModelWrapper(model)
    
    logits = model(input_ids).logits
    pred_class = torch.argmax(logits, dim=-1).item()
    
    # LayerIntegratedGradients on the word embeddings
    lig = LayerIntegratedGradients(wrapper, model.get_input_embeddings())
    
    attributions, delta = lig.attribute(inputs=input_ids, target=pred_class, return_convergence_delta=True)
    
    # Sum over embedding dimension
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / (torch.norm(attributions) + 1e-9)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, attributions.cpu().numpy(), pred_class

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {
        'ModernBERT': 'experiments/modernbert_runs/final_model',
        'CryptoBERT': 'experiments/runs/cryptobert/final_model'
    }
    
    results = []
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path): continue
        print(f"Generating attributions for {model_name}...")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.eval()
            
            for text in XAI_SAMPLES:
                tokens, attr, pred = get_attributions(text, model, tokenizer, device)
                top_indices = np.argsort(np.abs(attr))[-5:][::-1]
                top_tokens = [(tokens[i], float(attr[i])) for i in top_indices]
                
                results.append({
                    'Model': model_name,
                    'Text': text,
                    'Predicted': pred,
                    'TopTokens': top_tokens
                })
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    res_df = pd.DataFrame(results)
    os.makedirs('research/results/xai', exist_ok=True)
    res_df.to_csv('research/results/xai/xai_results.csv', index=False)
    
    with open('research/results/xai/xai_report.md', 'w') as f:
        f.write("# Interpretability Case Study (Integrated Gradients)\n\n")
        f.write("Comparing token-level attributions for key crypto-sentiment patterns.\n\n")
        for i, row in res_df.iterrows():
            f.write(f"### {row['Model']} | Sample {i%len(XAI_SAMPLES)}\n")
            f.write(f"**Text**: `{row['Text']}`\n")
            f.write(f"**Predicted Class**: `{row['Predicted']}`\n")
            f.write(f"**Top Attributions**:\n")
            for t, a in row['TopTokens']:
                f.write(f"- `{t:15}` : {a:.4f}\n")
            f.write("\n")

if __name__ == "__main__":
    run_experiment()
