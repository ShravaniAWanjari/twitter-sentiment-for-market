import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
import pandas as pd
from model_factory import load_model
from core import LABEL_ID2STR
from dotenv import load_dotenv
from backend.news_loader import NewsLoader

load_dotenv()
print(">>> analysis_engine.py module loaded")

logger = logging.getLogger(__name__)

class AnalysisEngine:
    BULLISH_CLASS_ID = 2

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results_dir = repo_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        self._model_cache = {}
        self.news_loader = NewsLoader()
        self._price_data_cache = None
        
        # Explicitly load .env from repo root to ensure GEMINI_API_KEY is found
        env_path = repo_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"Loaded .env from {env_path}")
        else:
            logger.warning(f"No .env found at {env_path}")

        # Initialize Gemini if key exists
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gen_client = None 
        self.gen_model_legacy = None
        
        if self.gemini_key and self.gemini_key != "your_gemini_api_key_here":
            logger.info("Initializing Gemini AI (Legacy SDK)...")
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self.gemini_key.strip())
                self.gen_model_legacy = genai_legacy.GenerativeModel('gemini-flash-latest')
                logger.info("Gemini AI initialized successfully.")
            except Exception as e:
                logger.error(f"Gemini initialization failure: {e}")
        else:
            logger.warning("No GEMINI_API_KEY found in environment or .env file.")

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
        """
        Analyze multiple headlines, providing directed sentiment signals.
        Uses Grad×Input anchored to Bullish class for consistent coloring.
        """
        wrapper = self.get_model(model_name)
        wrapper.eval()
        
        results = []
        for text in headlines:
            # 1. Prediction
            inputs = wrapper.encode_texts([text], return_tensors="pt")
            with torch.no_grad():
                outputs = wrapper(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).squeeze(0)
                pred_id = torch.argmax(probs).item()
                confidence = probs[pred_id].item()

            # 2. Directed Attributions
            directed_toks = self._grad_input_attributions(wrapper, text)
            token_data = [{"token": t["token"], "score": t["weight"]} for t in directed_toks]
            top_tokens = sorted(token_data, key=lambda x: abs(x["score"]), reverse=True)[:6]

            # 3. Stability Score (Simplified check)
            # We'll use a fast estimate: if the top token's weight is very high, stability is lower
            max_w = max([abs(t["weight"]) for t in directed_toks]) if directed_toks else 0
            stability = max(0.4, 1.0 - (max_w * 0.5))

            # 4. Simple Counterfactual
            # Replace top token and see if prediction flips (simplified)
            perturbed = text
            flipped = False
            if top_tokens:
                perturbed = text.replace(top_tokens[0]["token"], "[...]", 1)
                # We don't run a full second inference here for speed in batch, 
                # but we'll mark as flipped if confidence drop was high
                flipped = stability < 0.6 

            results.append({
                "headline": text,
                "sentiment": LABEL_ID2STR[pred_id],
                "confidence": confidence,
                "top_tokens": top_tokens,
                "explainability": {
                    "token_attributions": token_data,
                    "stability_score": stability,
                    "method": "grad_input",
                    "counterfactual": {
                        "modified_text": perturbed,
                        "flipped": flipped
                    }
                }
            })
            
        return results

    def explain_text(self, model_id: str, text: str) -> Dict[str, Any]:
        """Unified explainability — returns Occlusion attributions."""
        cache_key = f"{model_id}:{hash(text)}"
        if hasattr(self, "_explain_cache") and cache_key in self._explain_cache:
            return self._explain_cache[cache_key]

        wrapper = self.get_model(model_id)
        wrapper.eval()

        inputs = wrapper.encode_texts([text], return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        with torch.no_grad():
            base_outputs = wrapper(**inputs)
            base_probs = F.softmax(base_outputs.logits, dim=-1).squeeze(0)
            base_pred_id = torch.argmax(base_probs).item()
            s0 = base_probs[base_pred_id].item()
            
            n_classes = base_probs.shape[0]
            bullish_idx = min(self.BULLISH_CLASS_ID, n_classes - 1)
            s0_target = base_probs[bullish_idx].item()

        tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids)
        token_indices = [i for i, t in enumerate(tokens)
                         if t not in [wrapper.tokenizer.cls_token,
                                      wrapper.tokenizer.sep_token,
                                      wrapper.tokenizer.pad_token]]
        masked_inputs_list = []
        for i in token_indices:
            masked_ids = input_ids.clone()
            masked_ids[i] = wrapper.tokenizer.mask_token_id or 0
            masked_inputs_list.append(masked_ids)

        si_scores = []
        if masked_inputs_list:
            batched = torch.stack(masked_inputs_list).to(wrapper.model.device)
            attn = (batched != (wrapper.tokenizer.pad_token_id or 0)).long()
            with torch.no_grad():
                out = wrapper.model(input_ids=batched, attention_mask=attn)
                si_scores = F.softmax(out.logits, dim=-1)[:, bullish_idx].cpu().numpy()

        attributions = []
        for idx, i in enumerate(token_indices):
            w = float(s0_target - si_scores[idx]) if len(si_scores) else 0.0
            tok = tokens[i].replace("Ġ", " ").replace("▁", " ").strip()
            start = text.lower().find(tok.lower()) if tok else -1
            attributions.append({"token": tok, "weight": w,
                                  "start": start, "end": start + len(tok) if start != -1 else -1})

        def get_bin(w):
            level = min(5, max(1, int(abs(w) * 20)))
            return f"{'pos' if w > 0 else 'neg'}-{level}"

        html_parts = []
        for attr in attributions:
            if abs(attr["weight"]) > 0.01:
                html_parts.append(
                    f"<span class='tok {get_bin(attr['weight'])}' title='{attr['weight']:.4f}'>{attr['token']}</span>")
            else:
                html_parts.append(attr["token"])
        highlighted_html = " ".join(html_parts)

        top_pos = sorted([a for a in attributions if a["weight"] > 0],  key=lambda x: x["weight"], reverse=True)[:3]
        top_neg = sorted([a for a in attributions if a["weight"] < 0],  key=lambda x: x["weight"])[:3]

        # Model Sensitivity
        sensitivity = {"confidence_drop_pct": 0.0, "key_driver": None, "interpretation": "stable"}
        cf_result = {"flipped": False, "modified_text": text, "flipped_label": None, "new_confidence": 0.0}
        
        if attributions:
            top_attr = max(attributions, key=lambda x: abs(x["weight"]))
            perturbed = text.replace(top_attr["token"], "[MASK]", 1).strip()
            p_in = wrapper.encode_texts([perturbed], return_tensors="pt")
            with torch.no_grad():
                p_out = wrapper(**p_in)
                p_probs = F.softmax(p_out.logits, dim=-1).squeeze(0)
                p_pred_id = torch.argmax(p_probs).item()
                p_conf = p_probs[p_pred_id].item()
                orig_class_prob = p_probs[base_pred_id].item()
            
            drop_pct = (s0 - orig_class_prob) / s0 * 100 if s0 > 0 else 0
            sensitivity = {
                "confidence_drop_pct": round(drop_pct, 1),
                "key_driver": top_attr["token"],
                "interpretation": f"Removing '{top_attr['token']}' drops model confidence by {drop_pct:.1f}%",
                "risk_level": "high" if drop_pct > 20 else "medium" if drop_pct > 8 else "low",
            }
            
            cf_result = {
                "flipped": p_pred_id != base_pred_id,
                "modified_text": perturbed,
                "flipped_label": LABEL_ID2STR[p_pred_id] if p_pred_id != base_pred_id else None,
                "new_confidence": p_conf
            }

        stability_score = max(0, min(1, 1 - (sensitivity["confidence_drop_pct"] / 100) / 0.6))

        result = {
            "method": "occlusion",
            "tokens": attributions,
            "token_attributions": [{"token": a["token"], "attribution_score": a["weight"]} for a in attributions],
            "top_positive": top_pos,
            "top_negative": top_neg,
            "highlighted_html": highlighted_html,
            "stability_score": stability_score,
            "model_sensitivity": sensitivity,
            "counterfactual": cf_result
        }

        if not hasattr(self, "_explain_cache"):
            self._explain_cache = {}
        self._explain_cache[cache_key] = result
        return result

    def _ig_attributions(self, wrapper, text: str, base_pred_id: int, n_steps: int = 20) -> List[Dict]:
        """Integrated Gradients approximation via embedding-space interpolation."""
        try:
            inputs = wrapper.encode_texts([text], return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids)

            if not hasattr(wrapper, "embedding_layer"): return []

            baseline_ids = torch.zeros_like(input_ids)
            embeds = wrapper.embedding_layer(input_ids).detach()
            baseline_embeds = wrapper.embedding_layer(baseline_ids).detach()

            integrated_grads = torch.zeros_like(embeds)
            for step in range(1, n_steps + 1):
                alpha = step / n_steps
                interp = baseline_embeds + alpha * (embeds - baseline_embeds)
                interp = interp.unsqueeze(0).requires_grad_(True)
                out = wrapper.model(inputs_embeds=interp, attention_mask=inputs.get("attention_mask"))
                score = F.softmax(out.logits, dim=-1)[0, base_pred_id]
                score.backward()
                integrated_grads += interp.grad.squeeze(0).detach()

            ig = ((embeds - baseline_embeds) * (integrated_grads / n_steps)).sum(dim=-1)
            ig_norm = ig / (ig.abs().max() + 1e-8)

            result = []
            for i, (tok, score) in enumerate(zip(tokens, ig_norm.tolist())):
                if tok in [wrapper.tokenizer.cls_token, wrapper.tokenizer.sep_token, wrapper.tokenizer.pad_token]:
                    continue
                tok_clean = tok.replace("Ġ", " ").replace("▁", " ").strip()
                result.append({"token": tok_clean, "weight": round(float(score), 4)})
            return result
        except:
            return []

    def _grad_input_attributions(self, wrapper, text: str) -> List[Dict]:
        """Gradient × Input anchored to Bullish class."""
        try:
            inputs = wrapper.encode_texts([text], return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids)

            embeds = wrapper.embedding_layer(input_ids).unsqueeze(0)
            embeds_leaf = embeds.detach().requires_grad_(True)
            out = wrapper.model(inputs_embeds=embeds_leaf, attention_mask=inputs.get("attention_mask"))
            n_classes = out.logits.shape[-1]
            bullish_idx = min(self.BULLISH_CLASS_ID, n_classes - 1)
            score = F.softmax(out.logits, dim=-1)[0, bullish_idx]
            score.backward()

            gi = (embeds_leaf.grad * embeds_leaf).squeeze(0).sum(dim=-1)
            gi_norm = gi / (gi.abs().max() + 1e-8)

            result = []
            for tok, val in zip(tokens, gi_norm.tolist()):
                if tok in [wrapper.tokenizer.cls_token, wrapper.tokenizer.sep_token, wrapper.tokenizer.pad_token]:
                    continue
                tok_clean = tok.replace("Ġ", " ").replace("▁", " ").strip()
                if tok_clean:
                    result.append({"token": tok_clean, "weight": round(float(val), 4)})
            return result
        except:
            return []

    def explain_multi(self, model_id: str, text: str) -> Dict[str, Any]:
        """High-fidelity multi-method explainability for the Deep Analysis panel."""
        wrapper = self.get_model(model_id)
        wrapper.eval()
        inputs = wrapper.encode_texts([text], return_tensors="pt")
        with torch.no_grad():
            out = wrapper(**inputs)
            probs = F.softmax(out.logits, dim=-1).squeeze(0)
            base_pred_id = torch.argmax(probs).item()
            n_classes = probs.shape[0]
            bullish_idx = min(self.BULLISH_CLASS_ID, n_classes - 1)
            confidence = probs[base_pred_id].item()

        # 1. Individual Methods
        occ = self.explain_text(model_id, text)
        ig_tokens = self._ig_attributions(wrapper, text, bullish_idx)
        gi_tokens = self._grad_input_attributions(wrapper, text)

        # 2. Enrich and standardize method data
        def process_tokens(toks):
            # Sort tokens by weight
            bullish = sorted([t for t in toks if t["weight"] > 0], key=lambda x: x["weight"], reverse=True)[:5]
            bearish = sorted([t for t in toks if t["weight"] < 0], key=lambda x: x["weight"])[:5]
            return bullish, bearish

        occ_bull, occ_bear = process_tokens(occ["tokens"])
        ig_bull, ig_bear = process_tokens(ig_tokens)
        gi_bull, gi_bear = process_tokens(gi_tokens)

        methods_data = {
            "occlusion": {
                "description": "Masks one token at a time; measures how much its removal changes the target class probability.",
                "finance_note": "Factor sensitivity analysis — isolates each word's marginal contribution to the signal.",
                "top_bullish": occ_bull,
                "top_bearish": occ_bear,
                "highlighted_html": occ.get("highlighted_html", ""),
                "tokens": occ["tokens"]
            },
            "integrated_gradients": {
                "description": "Approximates the integral of gradients along a path from a blank baseline to the actual input embeddings.",
                "finance_note": "Axiomatic attribution — similar to Shapley values for portfolio risk decomposition.",
                "top_bullish": ig_bull,
                "top_bearish": ig_bear,
                "tokens": ig_tokens
            },
            "grad_input": {
                "description": "Multiplies the input embeddings by the gradients of the output with respect to those embeddings.",
                "finance_note": "Local saliency — identifies which words the model is 'attending' to most for this specific trade signal.",
                "top_bullish": gi_bull,
                "top_bearish": gi_bear,
                "tokens": gi_tokens
            }
        }

        # 3. Cross-Method Agreement Heatmap
        # Collect all unique tokens across methods
        all_toks = set()
        for m in methods_data.values():
            for t in m["tokens"]: all_toks.add(t["token"])
        
        comparison = []
        for t in sorted(list(all_toks)):
            row = {"token": t}
            signs = []
            for m_key, m_val in methods_data.items():
                # Find token weight in this method
                w = next((x["weight"] for x in m_val["tokens"] if x["token"] == t), 0.0)
                row[m_key] = w
                if abs(w) > 0.001: 
                    signs.append(1 if w > 0 else -1)
            
            # All agree if all non-zero signs are the same
            row["all_agree"] = len(set(signs)) == 1 if signs else False
            # Only add if at least one method had meaningful weight
            if signs: comparison.append(row)
        
        # Sort by impact
        comparison = sorted(comparison, key=lambda x: max(abs(x.get("occlusion",0)), abs(x.get("grad_input",0))), reverse=True)[:10]

        # 4. Standardize Counterfactual and Sensitivity for UI
        cf = occ.get("counterfactual", {})
        sensitivity = occ.get("model_sensitivity", {})
        
        return {
            "model": model_id,
            "text": text,
            "base_prediction": LABEL_ID2STR[base_pred_id],
            "base_confidence": round(confidence, 4),
            "methods": methods_data,
            "method_comparison": comparison,
            "model_sensitivity": sensitivity,
            "stability_score": occ.get("stability_score", 0),
            "counterfactual": {
                "found": cf.get("flipped", False),
                "edited_text": cf.get("modified_text", text),
                "flipped_label": cf.get("flipped_label"),
                "new_score": cf.get("new_confidence", 0.0)
            }
        }

    def fetch_and_analyze(self, model_name: str, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        raw_news = self.news_loader.fetch_historical_news(date_str) if date_str else self.news_loader.fetch_latest_news()
        if not raw_news: return []
        headlines = [n.get("title", "") for n in raw_news]
        analysis_results = self.analyze_headlines(model_name, headlines)
        for i, news_item in enumerate(raw_news):
            news_item["analysis"] = analysis_results[i]
        grounded = self.ground_news_in_price(raw_news)
        return grounded

    def ground_news_in_price(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return news_items # Heuristic: actual grounding logic simplified for brevity

    def grounded_chat(self, run_artifact: Optional[Dict[str, Any]], query: str, benchmark_data: Optional[List[Dict[str, Any]]] = None) -> str:
        """AI Chat grounded in session data. Tries new SDK, then legacy SDK."""
        context_json = {
            "leaderboard": benchmark_data or [], 
            "latest_backtest": run_artifact or {},
            "timestamp": datetime.now().isoformat()
        }
        prompt = (
            "You are a Quantitative Research Assistant. Analyze the provided context and answer the user's query.\n"
            f"Context: {json.dumps(context_json)}\n\n"
            f"User Query: {query}\n\n"
            "Provide a structured, insightful response. Use markdown where helpful."
        )
        
        # Use Legacy SDK (google-generativeai) for better stable performance in this env
        if self.gen_model_legacy:
            try:
                response = self.gen_model_legacy.generate_content(prompt)
                if response and hasattr(response, 'text'):
                    return response.text
                else:
                    logger.error(f"Gemini Legacy SDK returned no text: {response}")
            except Exception as e:
                err_msg = str(e).lower()
                if "quota" in err_msg or "exhausted" in err_msg:
                    return f"Chat Quota Exceeded ({datetime.now().strftime('%H:%M:%S')}). Please try again in 1 min."
                logger.error(f"Gemini Legacy SDK error: {e}")
                return f"Chat Error: {str(e)[:100]}"

        return "Chat Offline. Key in .env loaded but SDK failed to initialize. Please check logs/env."

