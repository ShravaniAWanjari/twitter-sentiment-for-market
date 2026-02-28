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
from backend.news_loader import NewsLoader

load_dotenv()

logger = logging.getLogger(__name__)

class AnalysisEngine:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results_dir = repo_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        self._model_cache = {}
        self.news_loader = NewsLoader()
        self._price_data_cache = None
        
        # Initialize Gemini if key exists
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gen_client = None
        self.gen_model_legacy = None
        
        if self.gemini_key and self.gemini_key != "your_gemini_api_key_here":
            try:
                # Try new SDK first
                from google import genai
                self.gen_client = genai.Client(api_key=self.gemini_key)
                logger.info("Gemini AI (google-genai) initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize new Gemini SDK, trying legacy: {e}")
                try:
                    import google.generativeai as genai_legacy
                    genai_legacy.configure(api_key=self.gemini_key)
                    self.gen_model_legacy = genai_legacy.GenerativeModel('gemini-1.5-flash')
                    logger.info("Gemini AI (legacy) initialized.")
                except Exception as e2:
                    logger.error(f"Total Gemini initialization failure: {e2}")

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
            # We now use the unified explain logic for individual headlines if needed,
            # but for bulk analysis we keep it simple or call explain_text for each.
            inputs = wrapper.encode_texts([text], return_tensors="pt")
            with torch.no_grad():
                outputs = wrapper(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).squeeze(0)
                pred_id = torch.argmax(probs).item()
                confidence = probs[pred_id].item()

            # Original simple attention-based tokens for bulk view
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

    def explain_text(self, model_id: str, text: str) -> Dict[str, Any]:
        """Unified explainability endpoint with Occlusion, Stability, and Counterfactuals."""
        cache_key = f"{model_id}:{hash(text)}"
        if hasattr(self, "_explain_cache") and cache_key in self._explain_cache:
            return self._explain_cache[cache_key]

        wrapper = self.get_model(model_id)
        wrapper.eval()

        # 1. Base Score
        inputs = wrapper.encode_texts([text], return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        with torch.no_grad():
            base_outputs = wrapper(**inputs)
            base_logits = base_outputs.logits
            base_probs = F.softmax(base_logits, dim=-1).squeeze(0)
            base_pred_id = torch.argmax(base_probs).item()
            s0 = base_probs[base_pred_id].item()

        # 2. Occlusion Attribution (Mask-one-token)
        tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids)
        token_indices = [i for i, t in enumerate(tokens) if t not in [wrapper.tokenizer.cls_token, wrapper.tokenizer.sep_token, wrapper.tokenizer.pad_token]]
        
        # Batch masked inputs
        masked_inputs_list = []
        for i in token_indices:
            masked_ids = input_ids.clone()
            masked_ids[i] = wrapper.tokenizer.mask_token_id if wrapper.tokenizer.mask_token_id else 0 # Fallback to 0 if no mask token
            masked_inputs_list.append(masked_ids)
        
        if masked_inputs_list:
            batched_masked = torch.stack(masked_inputs_list).to(wrapper.model.device)
            # Create attention mask for batch
            attn_mask = (batched_masked != (wrapper.tokenizer.pad_token_id or 0)).long()
            
            with torch.no_grad():
                masked_outputs = wrapper.model(input_ids=batched_masked, attention_mask=attn_mask)
                masked_probs = F.softmax(masked_outputs.logits, dim=-1)
                si_scores = masked_probs[:, base_pred_id].cpu().numpy()
        else:
            si_scores = []

        attributions = []
        for idx, i in enumerate(token_indices):
            weight = s0 - si_scores[idx]
            token_clean = tokens[i].replace("Ġ", " ").replace(" ", " ").strip()
            # Find start/end in original text (simplified)
            start_pos = text.lower().find(token_clean.lower()) if token_clean else -1
            attributions.append({
                "token": token_clean,
                "weight": float(weight),
                "start": start_pos,
                "end": start_pos + len(token_clean) if start_pos != -1 else -1
            })

        # 3. Highlighted HTML (Simplified intensity binning)
        def get_bin(w):
            mag = abs(w)
            level = min(5, max(1, int(mag * 20))) # Scale 0.05 -> 1, 0.25 -> 5
            return f"{'pos' if w > 0 else 'neg'}-{level}"

        html_parts = []
        for attr in attributions:
            if abs(attr["weight"]) > 0.01:
                html_parts.append(f"<span class='tok {get_bin(attr['weight'])}' title='{attr['weight']:.4f}'>{attr['token']}</span>")
            else:
                html_parts.append(attr["token"])
        
        highlighted_html = " ".join(html_parts)

        # 4. Stability
        top_pos = sorted([a for a in attributions if a["weight"] > 0], key=lambda x: x["weight"], reverse=True)[:3]
        top_neg = sorted([a for a in attributions if a["weight"] < 0], key=lambda x: x["weight"])[:3]
        
        perturbations = []
        # Drop top token
        if attributions:
            top_attr = max(attributions, key=lambda x: abs(x["weight"]))
            perturbed_text = text.replace(top_attr["token"], "", 1).strip()
            p_inputs = wrapper.encode_texts([perturbed_text], return_tensors="pt")
            with torch.no_grad():
                p_outputs = wrapper(**p_inputs)
                p_probs = F.softmax(p_outputs.logits, dim=-1).squeeze(0)
                p_score = p_probs[base_pred_id].item()
            perturbations.append({
                "type": "drop_top_token",
                "old_score": s0,
                "new_score": p_score,
                "diff": p_score - s0
            })

        # Stability score calculation
        mean_abs_delta = sum(abs(p["new_score"] - p["old_score"]) for p in perturbations) / len(perturbations) if perturbations else 0
        stability_score = max(0, min(1, 1 - mean_abs_delta / 0.6))

        # 5. Counterfactual (Antonym flip)
        antonyms = {"surges": "plummets", "adoption": "rejection", "illegal": "legal", "crackdown": "support", "nervous": "optimistic", "breaks": "falls", "surge": "drop", "up": "down", "positive": "negative"}
        cf_result = {"found": False}
        for attr in top_pos + top_neg:
            word = attr["token"].lower()
            if word in antonyms:
                cf_text = text.replace(attr["token"], antonyms[word], 1)
                cf_inputs = wrapper.encode_texts([cf_text], return_tensors="pt")
                with torch.no_grad():
                    cf_outputs = wrapper(**cf_inputs)
                    cf_probs = F.softmax(cf_outputs.logits, dim=-1).squeeze(0)
                    cf_pred_id = torch.argmax(cf_probs).item()
                    if cf_pred_id != base_pred_id:
                        cf_result = {
                            "found": True,
                            "edited_text": cf_text,
                            "old_score": s0,
                            "new_score": cf_probs[cf_pred_id].item(),
                            "flipped_label": LABEL_ID2STR[cf_pred_id]
                        }
                        break

        result = {
            "method": "occlusion",
            "tokens": attributions,
            "top_positive": top_pos,
            "top_negative": top_neg,
            "highlighted_html": highlighted_html,
            "stability": {
                "score_0_1": stability_score,
                "label": "high" if stability_score > 0.8 else "medium" if stability_score > 0.4 else "low",
                "perturbations": perturbations
            },
            "counterfactual": cf_result
        }

        # Cache it
        if not hasattr(self, "_explain_cache"):
            self._explain_cache = {}
        self._explain_cache[cache_key] = result
        return result

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

    def grounded_chat(self, run_artifact: Optional[Dict[str, Any]], query: str, benchmark_data: Optional[List[Dict[str, Any]]] = None) -> str:
        prompt = f"""
        You are a financial research assistant grounded in model benchmarking and backtest simulation data.
        
        ### Accuracy Overview & Leaderboard
        {json.dumps(benchmark_data) if benchmark_data else "No benchmark data available."}
        
        ### Latest Backtest Simulation
        {json.dumps(run_artifact) if run_artifact else "No backtest data available yet."}
        
        User Question: {query}
        
        Rules:
        1. Stay strictly grounded in the provided JSON data.
        2. Provide research-focused insights comparing model accuracy, latency, and backtest returns.
        3. If asked for advice, pivot to explaining the metrics and alpha performance.
        4. Include a standard disclaimer: "This analysis is for research purposes only and does not constitute financial advice."
        5. Be concise and professional.
        """

        if self.gen_client:
            try:
                # Try 2.0 first, then 1.5
                for m_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
                    try:
                        response = self.gen_client.models.generate_content(
                            model=m_name,
                            contents=prompt
                        )
                        return response.text
                    except Exception as inner_e:
                        logger.debug(f"Model {m_name} failed: {inner_e}")
                        continue
            except Exception as e:
                logger.error(f"New Gemini SDK failed: {e}")

        if self.gen_model_legacy:
            try:
                response = self.gen_model_legacy.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Legacy Gemini SDK failed: {e}")
        
        return self._fallback_chat(run_artifact, query, benchmark_data=benchmark_data)

    def _fallback_chat(self, run_artifact: Optional[Dict[str, Any]], query: str, benchmark_data: Optional[List[Dict[str, Any]]] = None) -> str:
        """Heuristic-based fallback if Gemini is unavailable."""
        msg = "I'm currently operating in fallback mode (Gemini offline). "
        
        if benchmark_data:
            # Find model with best f1_macro
            best_model = max(benchmark_data, key=lambda x: x.get("f1_macro", 0))
            msg += f"Based on the Leaderboard, {best_model.get('model')} has the highest F1 Macro of {best_model.get('f1_macro', 0):.4f}. "
        
        if run_artifact:
            metrics = run_artifact.get("metrics", {}).get("gated", {})
            total_ret = metrics.get("total_return", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            msg += (
                f"In the latest backtest for {run_artifact.get('model', 'the model')}, "
                f"the strategy achieved a total return of {total_ret:.2%} and a Sharpe Ratio of {sharpe:.2f}. "
            )
        
        msg += "This analysis is for research purposes only."
        return msg

    def ground_news_in_price(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate news items with BTC price action from Jan 2025 tick data."""
        if self._price_data_cache is None:
            try:
                from backend.backtest_engine import BacktestEngine
                engine = BacktestEngine(self.repo_root)
                df = engine.load_price_data()
                if not df.empty:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    # Ensure it's sorted
                    df = df.sort_index()
                    self._price_data_cache = df
                    logger.info("Price data cached for news grounding.")
            except Exception as e:
                logger.error(f"Failed to load price data for grounding: {e}")
                return news_items

        df = self._price_data_cache
        if df is None or df.empty:
            return news_items

        import pandas as pd
        grounded_results = []
        for item in news_items:
            try:
                pub_date = pd.to_datetime(item.get("pubDate"))
                # Localize or convert to match index if needed, assuming UTC
                if pub_date.tzinfo:
                    pub_date = pub_date.tz_convert(None)
                
                # Find nearest price at t0
                idx = df.index.get_indexer([pub_date], method='nearest')[0]
                price_t0 = df.iloc[idx]["Close"]
                
                # Find impact after 4 hours
                t4 = pub_date + pd.Timedelta(hours=4)
                idx4 = df.index.get_indexer([t4], method='nearest')[0]
                price_t4 = df.iloc[idx4]["Close"]
                
                delta_4h = (price_t4 - price_t0) / price_t0
                
                item["price_grounding"] = {
                    "t0_price": float(price_t0),
                    "t4_price": float(price_t4),
                    "delta_4h": float(delta_4h),
                    "impact": "bullish" if delta_4h > 0.005 else "bearish" if delta_4h < -0.005 else "neutral"
                }
            except Exception as e:
                logger.warning(f"Grounding failed for item: {e}")
                item["price_grounding"] = None
            
            grounded_results.append(item)
            
        return grounded_results

    def fetch_and_analyze(self, model_name: str, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch news from API, analyze sentiment, and ground in price data."""
        if date_str:
            raw_news = self.news_loader.fetch_historical_news(date_str)
        else:
            raw_news = self.news_loader.fetch_latest_news()
            
        if not raw_news:
            return []
            
        # Analyze sentiment for each
        headlines = [n.get("title", "") for n in raw_news]
        analysis_results = self.analyze_headlines(model_name, headlines)
        
        # Merge results
        for i, news_item in enumerate(raw_news):
            news_item["analysis"] = analysis_results[i]
            
        # Ground in price
        grounded = self.ground_news_in_price(raw_news)
        return grounded
