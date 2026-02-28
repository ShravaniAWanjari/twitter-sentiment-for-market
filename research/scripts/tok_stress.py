
import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Slang terms to analyze
SLANG_TERMS = [
    "hodl", "rekt", "moon", "mooning", "pump", "dump", "bulls", "bears",
    "ath", "fomo", "fud", "sats", "fiat", "whale", "bagholder",
    "cryptocurrency", "blockchain", "ethereum", "stablecoin", "halving",
    "altcoin", "to the moon", "diamond hands", "rugged", "paper hands"
]

def analyze_tokenizer(name, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    stats = []
    
    for term in SLANG_TERMS:
        tokens = tokenizer.tokenize(term)
        fragmentation = len(tokens)
        stats.append({
            'Model': name,
            'Term': term,
            'Tokens': [t.encode('ascii', errors='replace').decode('ascii') for t in tokens],
            'FragCount': fragmentation
        })
    return stats

def run_experiment():
    tokenizers = {
        'ModernBERT': 'answerdotai/ModernBERT-base',
        'CryptoBERT': 'ElKulako/cryptobert',
        'FinBERT': 'ProsusAI/finbert',
        'BERT-base': 'bert-base-uncased'
    }
    
    all_stats = []
    for name, model_id in tokenizers.items():
        print(f"Analyzing tokenizer: {name}")
        try:
            all_stats.extend(analyze_tokenizer(name, model_id))
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
        
    df = pd.DataFrame(all_stats)
    os.makedirs('research/results/tok_stress', exist_ok=True)
    df.to_csv('research/results/tok_stress/fragmentation.csv', index=False)
    
    summary = df.groupby('Model')['FragCount'].agg(['mean', 'max', 'std']).reset_index()
    top_frag = df.sort_values(['Model', 'FragCount'], ascending=[True, False]).groupby('Model').head(10)
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Option A: Boxplot + Jittered points
    sns.boxplot(x='Model', y='FragCount', data=df, palette="Pastel1", showfliers=False, width=0.5)
    sns.stripplot(x='Model', y='FragCount', data=df, color="black", alpha=0.4, jitter=0.2, size=5)
    
    plt.title("Token Fragmentation Counts for Crypto Slang (Boxplot + Jitter)", fontsize=15, pad=20)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylabel("Number of Wordpieces (Fragments)", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('research/results/tok_stress/frag_boxplot.png', dpi=300)
    plt.close()
    
    with open('research/results/tok_stress/tok_report.md', 'w', encoding='utf-8') as f:
        f.write("# Tokenization Stress Test Report\n\n")
        f.write("## Fragmentation Statistics (Slang Terms)\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Top fragmented terms per model\n\n")
        f.write(top_frag[['Model', 'Term', 'Tokens', 'FragCount']].to_markdown(index=False))

if __name__ == "__main__":
    run_experiment()
