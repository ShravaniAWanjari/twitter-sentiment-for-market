import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from model_factory import load_model
from core import LABEL_ID2STR
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class AnalysisEngine:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results_dir = repo_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        self._model_cache = {}
        
        # Initialize Gemini if key exists
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gen_model = None
        if self.gemini_key and self.gemini_key != "your_gemini_api_key_here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                # Use flash for speed (free tier)
                self.gen_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini AI initialized for grounded chat.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")

    def get_model(self, model_name: str):
        if model_name not in self._model_cache:
            model_dir = self.repo_root / "experiments" / "runs" / model_name / "final_model"
            if not model_dir.exists():
                logger.info(f"Local model {model_name} not found at {model_dir}, loading default.")
                self._model_cache[model_name] = load_model(model_name)
            else:
                self._model_cache[model_name] = load_model(model_name, model_name=str(model_dir))
        return self._model_cache[model_name]

    def analyze_headlines(self, model_name: str, headlines: List[str]) -> List[Dict[str, Any]]:
        wrapper = self.get_model(model_name)
        wrapper.eval()
        
        results = []
        for text in headlines:
            inputs = wrapper.encode_texts([text], return_tensors="pt")
            with torch.no_grad():
                outputs = wrapper(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).squeeze(0)
                pred_id = torch.argmax(probs).item()
                confidence = probs[pred_id].item()

            attentions = outputs.attentions 
            last_layer_attn = attentions[-1].squeeze(0).mean(dim=0)
            cls_attn = last_layer_attn[0, :].cpu().numpy()
            
            tokens = wrapper.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            cls_attn = cls_attn / (cls_attn.max() + 1e-9)
            
            token_data = []
            for token, score in zip(tokens, cls_attn):
                if token in [wrapper.tokenizer.cls_token, wrapper.tokenizer.sep_token, wrapper.tokenizer.pad_token]:
                    continue
                token_data.append({"token": token, "score": float(score)})

            results.append({
                "headline": text,
                "sentiment": LABEL_ID2STR[pred_id],
                "confidence": confidence,
                "tokens": token_data,
                "top_tokens": sorted(token_data, key=lambda x: x["score"], reverse=True)[:5]
            })
            
        return results

    def generate_pdf_report(self, run_artifact: Dict[str, Any]) -> Path:
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Quant NLP Research - Backtest Analysis Report", 0, 1, "C")
            pdf.ln(10)
            
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Model: {run_artifact.get('model')}", 0, 1)
            pdf.cell(0, 10, f"Strategy: {run_artifact.get('strategy')}", 0, 1)
            pdf.cell(0, 10, f"Timestamp: {run_artifact.get('timestamp')}", 0, 1)
            
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Strategy Metrics", 0, 1)
            pdf.set_font("Arial", "", 10)
            
            metrics = run_artifact.get("metrics", {})
            for stype, m in metrics.items():
                pdf.cell(0, 8, f"{stype.upper()}: Ret={m.get('total_return',0):.2%}, Sharpe={m.get('sharpe_ratio',0):.2f}", 0, 1)

            pdf_path = self.results_dir / "Analysis_Report.pdf"
            pdf.output(str(pdf_path))
            return pdf_path
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            report_path = self.results_dir / "Analysis_Report.txt"
            with open(report_path, "w") as f:
                f.write(json.dumps(run_artifact, indent=2))
            return report_path

    def grounded_chat(self, run_artifact: Dict[str, Any], query: str) -> str:
        query_lower = query.lower()
        if any(x in query_lower for x in ["buy", "sell", "invest", "financial advice", "should i"]):
            return "I am an AI research assistant and cannot provide financial advice."

        if self.gen_model:
            try:
                prompt = f"""
                You are a financial research assistant grounded in backtest data.
                Data context (JSON): {json.dumps(run_artifact)}
                User Question: {query}
                
                Rules:
                1. Stay strictly grounded in the provided JSON data.
                2. Be concise.
                3. Refuse all forms of financial advice.
                4. Focus on explaining performance metrics and NLP-gating alpha.
                """
                response = self.gen_model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                return "Gemini is currently unavailable. " + self._fallback_chat(run_artifact, query)
        
        return self._fallback_chat(run_artifact, query)

    def _fallback_chat(self, run_artifact: Dict[str, Any], query: str) -> str:
        # Simple deterministic logic if Gemini is off
        query_lower = query.lower()
        metrics = run_artifact.get("metrics", {})
        gated = metrics.get("gated", {})
        baseline = metrics.get("baseline", {})
        
        if "performance" in query_lower:
            diff = gated.get("total_return", 0) - baseline.get("total_return", 0)
            return f"The NLP-gated strategy {'outperformed' if diff > 0 else 'underperformed'} baseline by {abs(diff):.2%}. Gated Sharpe: {gated.get('sharpe_ratio', 0):.2f}."
        
        return "I'm currently in fallback mode. I can tell you about 'performance' using the latest backtest data."
