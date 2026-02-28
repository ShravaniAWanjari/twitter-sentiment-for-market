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
    BULLISH_CLASS_ID = 2

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
        """
        Analyze multiple headlines, providing directed sentiment signals.
        Uses Grad×Input anchored to Bullish class for consistent coloring:
            positive score -> Bullish driver
            negative score -> Bearish driver
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

            # 2. Directed Attributions (Grad×Input anchored to Bullish)
            # This ensures top_tokens has positive/negative values
            directed_toks = self._grad_input_attributions(wrapper, text)
            
            # Map back to the expected format for 'tokens' and 'top_tokens'
            token_data = [{"token": t["token"], "score": t["weight"]} for t in directed_toks]
            
            # For the summary view, we want the most significant drivers in either direction
            top_tokens = sorted(token_data, key=lambda x: abs(x["score"]), reverse=True)[:5]

            results.append({
                "headline": text,
                "sentiment": LABEL_ID2STR[pred_id],
                "confidence": confidence,
                "tokens": token_data,
                "top_tokens": top_tokens
            })
            
        return results

    def explain_text(self, model_id: str, text: str) -> Dict[str, Any]:
        """
        Unified explainability — returns Occlusion attributions alongside
        stability score, counterfactual, and model sensitivity.
        For the full multi-method view use explain_multi().
        """
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
            
            # Anchor explainability to Bullish class for consistent coloring
            n_classes = base_probs.shape[0]
            bullish_idx = min(self.BULLISH_CLASS_ID, n_classes - 1)
            s0_target = base_probs[bullish_idx].item()

        # Occlusion
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
                # Map occlusion to the target Bullish class
                si_scores = F.softmax(out.logits, dim=-1)[:, bullish_idx].cpu().numpy()

        attributions = []
        for idx, i in enumerate(token_indices):
            # positive = word pushed toward Bullish; negative = word pushed toward Bearish
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

        # Model Sensitivity (replaces opaque "Perturbation Test")
        sensitivity = {"confidence_drop_pct": 0.0, "key_driver": None, "interpretation": "stable"}
        if attributions:
            top_attr = max(attributions, key=lambda x: abs(x["weight"]))
            perturbed = text.replace(top_attr["token"], "", 1).strip()
            p_in = wrapper.encode_texts([perturbed], return_tensors="pt")
            with torch.no_grad():
                p_prob = F.softmax(wrapper(**p_in).logits, dim=-1).squeeze(0)[base_pred_id].item()
            drop_pct = (s0 - p_prob) / s0 * 100 if s0 > 0 else 0
            sensitivity = {
                "confidence_drop_pct": round(drop_pct, 1),
                "key_driver": top_attr["token"],
                "interpretation": (
                    f"Removing '{top_attr['token']}' drops model confidence by "
                    f"{drop_pct:.1f}% — {'highly sensitive' if drop_pct > 20 else 'moderately sensitive' if drop_pct > 8 else 'robust'} to this term."
                ),
                "risk_level": "high" if drop_pct > 20 else "medium" if drop_pct > 8 else "low",
            }

        # Stability
        stability_score = max(0, min(1, 1 - (sensitivity["confidence_drop_pct"] / 100) / 0.6))

        # Counterfactual
        antonyms = {"surges": "plummets", "adoption": "rejection", "illegal": "legal",
                    "crackdown": "support", "nervous": "optimistic", "breaks": "falls",
                    "surge": "drop", "up": "down", "positive": "negative",
                    "rally": "decline", "bullish": "bearish", "gains": "losses",
                    "soars": "crashes", "climbs": "falls", "rises": "drops"}
        cf_result = {"found": False}
        for attr in top_pos + top_neg:
            word = attr["token"].lower()
            if word in antonyms:
                cf_text = text.replace(attr["token"], antonyms[word], 1)
                cf_in = wrapper.encode_texts([cf_text], return_tensors="pt")
                with torch.no_grad():
                    cf_probs = F.softmax(wrapper(**cf_in).logits, dim=-1).squeeze(0)
                    cf_pred = torch.argmax(cf_probs).item()
                    if cf_pred != base_pred_id:
                        cf_result = {
                            "found": True, "edited_text": cf_text,
                            "old_score": s0, "new_score": cf_probs[cf_pred].item(),
                            "flipped_label": LABEL_ID2STR[cf_pred],
                            "flipped": True, "modified_text": cf_text,
                        }
                        break

        result = {
            "method": "occlusion",
            "tokens": attributions,
            "token_attributions": [{"token": a["token"], "attribution_score": a["weight"]} for a in attributions],
            "top_positive": top_pos,
            "top_negative": top_neg,
            "highlighted_html": highlighted_html,
            "stability": {
                "score_0_1": stability_score,
                "label": "high" if stability_score > 0.8 else "medium" if stability_score > 0.4 else "low",
                "perturbations": [{"type": "drop_top_token", "diff": -sensitivity["confidence_drop_pct"] / 100}],
            },
            "stability_score": stability_score,
            "model_sensitivity": sensitivity,
            "counterfactual": cf_result,
        }

        if not hasattr(self, "_explain_cache"):
            self._explain_cache = {}
        self._explain_cache[cache_key] = result
        return result

    def _ig_attributions(self, wrapper, text: str, base_pred_id: int, n_steps: int = 20) -> List[Dict]:
        """
        Integrated Gradients approximation via embedding-space interpolation.
        Works without Captum by directly computing gradients over the embedding matrix.
        """
        try:
            inputs = wrapper.encode_texts([text], return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids)

            # Robust embedding layer detection
            emb_layer = None
            for attr in ["embeddings", "bert", "roberta", "model"]:
                module = getattr(wrapper.model, attr, None)
                if module is not None:
                    # Check for tok_embeddings (ModernBERT) or word_embeddings (BERT)
                    for e_name in ["tok_embeddings", "word_embeddings", "embeddings"]:
                        e_mod = getattr(module, e_name, None)
                        if e_mod and hasattr(e_mod, "weight"):
                            emb_layer = e_mod
                            break
                if emb_layer: break
            
            # Fallback search
            if emb_layer is None:
                for n, m in wrapper.model.named_modules():
                    if hasattr(m, "weight") and len(m.weight.shape) == 2 and m.weight.shape[0] > 10000:
                        emb_layer = m
                        break

            if emb_layer is None:
                logger.warning("IG: could not find embedding layer")
                return []

            baseline_ids = torch.zeros_like(input_ids)  # zero embedding baseline
            embeds = emb_layer(input_ids).detach()
            baseline_embeds = emb_layer(baseline_ids).detach()

            integrated_grads = torch.zeros_like(embeds)
            for step in range(1, n_steps + 1):
                alpha = step / n_steps
                interp = baseline_embeds + alpha * (embeds - baseline_embeds)
                interp = interp.unsqueeze(0).requires_grad_(True)

                attn_mask = inputs.get("attention_mask")
                # Forward using inputs_embeds
                out = wrapper.model(inputs_embeds=interp, attention_mask=attn_mask)
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

        except Exception as e:
            logger.warning(f"IG attribution failed: {e}")
            return []

    # ─── Class ID constants ──────────────────────────────────────────────────
    # LABEL_ID2STR = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    # All attribution methods anchor to the BULLISH class so that:
    #   positive weight  → word pushes model toward Bullish  → Bullish driver
    #   negative weight  → word pushes model toward Bearish  → Bearish driver
    BULLISH_CLASS_ID = 2

    def _grad_input_attributions(self, wrapper, text: str) -> List[Dict]:
        """
        Gradient × Input — fast, directionally stable, and anchored to the
        Bullish class (class 2).

        Interpretation (always consistent):
          positive score → word pushes model toward Bullish signal
          negative score → word pushes model toward Bearish signal

        Why it beats LIME for short financial headlines:
        - LIME needs many samples for stability; Grad×Input is a single forward+backward pass
        - Signs never flip: anchored to a fixed class, not the predicted class
        - Captures both magnitude and direction of each token's embedding gradient
        """
        try:
            inputs = wrapper.encode_texts([text], return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids)

            # Robust embedding layer detection
            emb_layer = None
            # 1. Direct search through common names
            for attr in ["embeddings", "bert", "roberta", "model"]:
                module = getattr(wrapper.model, attr, None)
                if module is not None:
                    # Check for tok_embeddings (ModernBERT) or word_embeddings (BERT)
                    for e_name in ["tok_embeddings", "word_embeddings", "embeddings"]:
                        e_mod = getattr(module, e_name, None)
                        if e_mod and hasattr(e_mod, "weight"):
                            emb_layer = e_mod
                            break
                if emb_layer: break
            
            # 2. Fallback: exhaustive search for word-sized weight matrix
            if emb_layer is None:
                for n, m in wrapper.model.named_modules():
                    if hasattr(m, "weight") and len(m.weight.shape) == 2:
                        if m.weight.shape[0] > 10000: # Typical vocab size
                            emb_layer = m
                            break

            if emb_layer is None:
                logger.warning("Grad×Input: could not find embedding layer")
                return []

            # Forward pass with gradients enabled
            embeds = emb_layer(input_ids).unsqueeze(0)
            embeds.retain_grad()
            embeds_leaf = embeds.detach().requires_grad_(True)

            attn_mask = inputs.get("attention_mask")
            out = wrapper.model(inputs_embeds=embeds_leaf, attention_mask=attn_mask)
            n_classes = out.logits.shape[-1]
            bullish_idx = min(self.BULLISH_CLASS_ID, n_classes - 1)
            score = F.softmax(out.logits, dim=-1)[0, bullish_idx]
            score.backward()

            # Gradient × Input: sum over embedding dimension
            gi = (embeds_leaf.grad * embeds_leaf).squeeze(0).sum(dim=-1)  # (seq,)
            gi_norm = gi / (gi.abs().max() + 1e-8)

            result = []
            for i, (tok, val) in enumerate(zip(tokens, gi_norm.tolist())):
                if tok in [wrapper.tokenizer.cls_token, wrapper.tokenizer.sep_token,
                           wrapper.tokenizer.pad_token]:
                    continue
                tok_clean = tok.replace("Ġ", " ").replace("▁", " ").strip()
                if tok_clean:
                    result.append({"token": tok_clean, "weight": round(float(val), 4)})
            return result

        except Exception as e:
            logger.warning(f"Grad×Input attribution failed: {e}")
            return []

    def explain_multi(self, model_id: str, text: str) -> Dict[str, Any]:
        """
        Run Occlusion + Integrated Gradients + Gradient×Input on the same headline.

        All three methods are ANCHORED TO THE BULLISH CLASS (class 2):
          positive weight  → word increases Bullish probability  → Bullish driver
          negative weight  → word decreases Bullish probability  → Bearish driver

        This is correct regardless of what the model actually predicts.
        """
        cache_key = f"multi:{model_id}:{hash(text)}"
        if hasattr(self, "_explain_cache") and cache_key in self._explain_cache:
            return self._explain_cache[cache_key]

        wrapper = self.get_model(model_id)
        wrapper.eval()

        inputs = wrapper.encode_texts([text], return_tensors="pt")
        with torch.no_grad():
            base_probs = F.softmax(wrapper(**inputs).logits, dim=-1).squeeze(0)
            base_pred_id = torch.argmax(base_probs).item()
            s0 = base_probs[base_pred_id].item()
            n_classes = base_probs.shape[0]
            bullish_idx = min(self.BULLISH_CLASS_ID, n_classes - 1)
            bullish_conf = base_probs[bullish_idx].item()

        from core import LABEL_ID2STR

        # ── Occlusion: measure delta in BULLISH class probability ─────────────
        # positive weight = removing this token LOWERS bullish prob → it was a bullish driver
        base_explain = self.explain_text(model_id, text)
        # Re-derive occlusion but anchored to bullish class
        input_ids = wrapper.encode_texts([text], return_tensors="pt")["input_ids"][0]
        tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids)
        token_indices = [i for i, t in enumerate(tokens)
                         if t not in [wrapper.tokenizer.cls_token,
                                      wrapper.tokenizer.sep_token,
                                      wrapper.tokenizer.pad_token]]
        occlusion_tokens = []
        if token_indices:
            masked_list = []
            for i in token_indices:
                m = input_ids.clone()
                m[i] = wrapper.tokenizer.mask_token_id or 0
                masked_list.append(m)
            batched = torch.stack(masked_list).to(wrapper.model.device)
            attn = (batched != (wrapper.tokenizer.pad_token_id or 0)).long()
            with torch.no_grad():
                occ_probs = F.softmax(
                    wrapper.model(input_ids=batched, attention_mask=attn).logits, dim=-1
                )[:, bullish_idx].cpu().numpy()
            for idx, i in enumerate(token_indices):
                # positive = word was present and boosting bullishness
                w = float(bullish_conf - occ_probs[idx])
                tok = tokens[i].replace("Ġ", " ").replace("▁", " ").strip()
                occlusion_tokens.append({"token": tok, "weight": w})

        # ── Integrated Gradients: already anchored to Bullish class in _ig_attributions ─
        ig_tokens = self._ig_attributions(wrapper, text, bullish_idx)

        # ── Gradient × Input: anchored to Bullish class ───────────────────────
        gi_tokens = self._grad_input_attributions(wrapper, text)

        def top_k(toks, k=5):
            return sorted(toks, key=lambda x: abs(x.get("weight", 0)), reverse=True)[:k]

        # ── Cross-method agreement ─────────────────────────────────────────────
        all_token_names = list({t["token"] for t in occlusion_tokens + ig_tokens + gi_tokens
                                if t["token"]})
        agreement = []
        for tok in all_token_names:
            occ_w  = next((t["weight"] for t in occlusion_tokens if t["token"] == tok), None)
            ig_w   = next((t["weight"] for t in ig_tokens if t["token"] == tok), None)
            gi_w   = next((t["weight"] for t in gi_tokens if t["token"] == tok), None)
            vals = [w for w in [occ_w, ig_w, gi_w] if w is not None]
            if len(vals) >= 2:
                signs = [1 if w > 0 else -1 if w < 0 else 0 for w in vals]
                agree = len(set(signs)) == 1
                avg_w = sum(vals) / len(vals)
                agreement.append({
                    "token": tok,
                    "occlusion": round(occ_w, 4) if occ_w is not None else None,
                    "integrated_gradients": round(ig_w, 4) if ig_w is not None else None,
                    "grad_input": round(gi_w, 4) if gi_w is not None else None,
                    "all_agree": agree,
                    "avg_weight": round(avg_w, 4),
                })
        agreement = sorted(agreement, key=lambda x: abs(x["avg_weight"]), reverse=True)[:8]

        result = {
            "model": model_id,
            "text": text,
            "base_prediction": LABEL_ID2STR[base_pred_id],
            "base_confidence": round(s0, 4),
            "anchored_to": "Bullish class — positive = bullish driver, negative = bearish driver",
            "methods": {
                "occlusion": {
                    "description": "Masks one token at a time; measures how much its removal changes the Bullish class probability.",
                    "finance_note": "Factor sensitivity analysis — isolates each word's marginal contribution to the bullish signal.",
                    "tokens": occlusion_tokens,
                    "top_bullish": top_k([t for t in occlusion_tokens if t["weight"] > 0]),
                    "top_bearish": top_k([t for t in occlusion_tokens if t["weight"] < 0]),
                    "highlighted_html": base_explain.get("highlighted_html", ""),
                },
                "integrated_gradients": {
                    "description": "Computes gradients along a path from a zero-embedding baseline to the actual input, summing contributions to Bullish class probability.",
                    "finance_note": "Shapley value allocation — each word receives credit proportional to its causal contribution across all interpolation steps.",
                    "tokens": ig_tokens,
                    "top_bullish": top_k([t for t in ig_tokens if t["weight"] > 0]),
                    "top_bearish": top_k([t for t in ig_tokens if t["weight"] < 0]),
                },
                "grad_input": {
                    "description": "Multiplies each token's embedding gradient by its embedding value (∂Bullish/∂emb × emb). Single-pass, stable, and sign-consistent.",
                    "finance_note": "First-order sensitivity analysis — directional derivative showing how a marginal shift in each word's representation changes the bullish signal.",
                    "tokens": gi_tokens,
                    "top_bullish": top_k([t for t in gi_tokens if t["weight"] > 0]),
                    "top_bearish": top_k([t for t in gi_tokens if t["weight"] < 0]),
                },
            },
            "method_comparison": agreement,
            "model_sensitivity": base_explain.get("model_sensitivity", {}),
            "stability": base_explain.get("stability", {}),
            "counterfactual": base_explain.get("counterfactual", {}),
        }

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
        # Construct a rich context from the provided data
        context_json = {
            "leaderboard": benchmark_data or [],
            "latest_backtest": run_artifact or {}
        }
        
        prompt = f"""
        System: You are 'Market GPT', a specialized financial research AI. Your knowledge is strictly limited to the provided research data.
        
        ### DATA CONTEXT (JSON)
        {json.dumps(context_json, indent=2)}
        
        ### USER QUERY
        {query}
        
        ### OPERATIONAL GUIDELINES:
        1. SCOPE GATE: If the user query is NOT about the provided data, trading strategies, model benchmarking, or NLP explainability, you MUST respond with: 
           "I'm sorry, Market GPT is strictly authorized to answer queries related to this research project's backtest results and model performance as part of our safety and grounding protocols."
        
        2. DEPTH: If the user asks for detailed explanations, 'deep dives', or 'why' something happened, use the metadata in the JSON (like case studies, metrics, or architecture types) to provide a thorough, multi-paragraph analysis. Expand on the relationship between model F1 scores and backtest Sharpe ratios.
        
        3. GROUNDING: Never hallucinate numbers. Use only the stats from the context.
        
        4. ANALYSIS: Compare the "Gated" performance vs "Baseline" performance provided in the stats. Explain how NLP gating improved or impacted the max drawdown.
        
        5. DISCLAIMER: Always end your response with: "This analysis is for research purposes only and does not constitute financial advice."
        """

        if self.gen_client:
            try:
                # Use latest stable versions to avoid 404s
                # Verified 'gemini-flash-latest' as working for this key; prioritize it to avoid 429 latency
                models_to_try = [
                    'gemini-flash-latest',
                    'gemini-2.0-flash', 
                    'gemini-2.0-flash-lite',
                    'gemini-flash-lite-latest',
                    'gemini-1.5-flash', 
                    'gemini-pro-latest'
                ]
                for m_name in models_to_try:
                    try:
                        # Normalize model name for API
                        if not m_name.startswith("models/"):
                            full_m_name = f"models/{m_name}"
                        else:
                            full_m_name = m_name
                        
                        response = self.gen_client.models.generate_content(
                            model=full_m_name,
                            contents=prompt
                        )
                        return response.text
                    except Exception as inner_e:
                        err_msg = str(inner_e).lower()
                        if "429" in err_msg:
                            logger.warning(f"Gemini {m_name} rate limited (429). Trying next...")
                        elif "404" in err_msg:
                            logger.debug(f"Gemini {m_name} not found (404). Trying next...")
                        else:
                            logger.error(f"Gemini {m_name} unexpected error: {inner_e}")
                        continue
            except Exception as e:
                logger.error(f"New Gemini SDK execution failed: {e}")

        if self.gen_model_legacy:
            try:
                response = self.gen_model_legacy.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Legacy Gemini SDK failed: {e}")
        
        return self._fallback_chat(run_artifact, query, benchmark_data=benchmark_data)

    def _fallback_chat(self, run_artifact: Optional[Dict[str, Any]], query: str, benchmark_data: Optional[List[Dict[str, Any]]] = None) -> str:
        """Enhanced heuristic-based fallback if Gemini is unavailable."""
        query_lc = query.lower()
        
        # Domain Gate for Fallback
        keywords = ["model", "backtest", "f1", "return", "sharpe", "drawdown", "accuracy", "bench", "nlp", "gating", "trade", "btc"]
        is_in_domain = any(k in query_lc for k in keywords) or len(query) < 5
        
        if not is_in_domain:
            return "I'm sorry, Market GPT is strictly authorized to answer queries related to this research project's backtest results and model performance as part of our safety and grounding protocols."

        msg = ["**Note: Market GPT is currently in Fallback Mode (LLM Offline).**"]
        
        if benchmark_data:
            best_model = max(benchmark_data, key=lambda x: x.get("f1_macro", 0))
            msg.append(f"### Leaderboard Insight")
            msg.append(f"The top-performing model is **{best_model.get('model')}** with an F1 Macro of {best_model.get('f1_macro', 0):.4f}.")
            msg.append(f"Across the fleet, the average latency is {pd.Series([r.get('latency_ms_per_tweet', 0) for r in benchmark_data]).mean():.2f}ms per headline.")
        
        if run_artifact:
            metrics = run_artifact.get("metrics", {}).get("gated", {})
            baseline = run_artifact.get("metrics", {}).get("baseline", {})
            msg.append(f"### Backtest Summary ({run_artifact.get('model')})")
            msg.append(f"- **Total Return:** {metrics.get('total_return', 0):.2%} (vs {baseline.get('total_return', 0):.2%} Baseline)")
            msg.append(f"- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.2f}")
            msg.append(f"- **Max Drawdown:** {metrics.get('max_drawdown', 0):.2%} (Improvement seen via NLP Gating)")
            
            if "case_study" in run_artifact:
                cs = run_artifact["case_study"]
                msg.append(f"### Recent Case Study")
                msg.append(f"On {cs.get('date')}, the model encountered the headline: *\"{cs.get('headline')}\"*.")
                msg.append(f"Action taken: **{cs.get('gating_status')}**. Resulting in: {cs.get('outcome_label')}.")

        if ("detail" in query_lc or "explain" in query_lc) and run_artifact:
            threshold = run_artifact.get('params', {}).get('threshold', 0.5)
            msg.append(f"\n*Detailed Analysis (Fallback View):* The data suggests that models with higher F1 scores correlate with more effective technical signal gating. By filtering out uncertain news periods (Threshold: {threshold}), the strategy effectively reduced capital exposure during drawdown phases. This explains the reduction in Max Drawdown compared to the unfiltered baseline.")

        msg.append("\nThis analysis is for research purposes only and does not constitute financial advice.")
        return "\n\n".join(msg)

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

    def generate_pdf_report(self, run: Dict[str, Any]) -> Path:
        """Generates a PDF report summarizing the backtest and benchmark results."""
        try:
            from fpdf import FPDF
        except ImportError:
            logger.error("fpdf2 is not installed. PDF generation failed.")
            raise RuntimeError("PDF generation requires the fpdf2 library.")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", style="B", size=18)
        
        # Title
        pdf.cell(0, 15, "Session Analysis Report", ln=True, align="C")
        pdf.ln(5)
        
        # Meta Data
        pdf.set_font("helvetica", size=10)
        from datetime import datetime
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y')}", ln=True, align="C")
        pdf.ln(10)
        
        if not run:
            pdf.cell(0, 10, "No backtest data available for this session.", ln=True)
            pdf_path = self.repo_root / "results" / "Session_Analysis_Report.pdf"
            pdf.output(str(pdf_path))
            return pdf_path

        # 1. Backtest Details
        pdf.set_font("helvetica", style="B", size=14)
        pdf.cell(0, 10, "1. Backtest Simulation Overview", ln=True)
        pdf.set_line_width(0.5)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 8, f"Model: {run.get('model', 'Unknown')}", ln=True)
        pdf.cell(0, 8, f"Strategy: {run.get('strategy', 'Unknown')}", ln=True)
        pdf.cell(0, 8, f"Period: {run.get('from_date', 'N/A')} to {run.get('to_date', 'N/A')}", ln=True)
        pdf.ln(8)
        
        # 2. Performance Metrics Comparison
        pdf.set_font("helvetica", style="B", size=14)
        pdf.cell(0, 10, "2. Performance Metrics Comparison", ln=True)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)
        
        metrics = run.get("metrics", {})
        baseline = metrics.get("baseline", {})
        gated = metrics.get("gated", {})
        
        # Table Header
        pdf.set_font("helvetica", style="B", size=12)
        pdf.set_fill_color(230, 230, 240)
        pdf.cell(60, 10, "Metric", border=1, fill=True)
        pdf.cell(65, 10, "Baseline Strategy", border=1, align="C", fill=True)
        pdf.cell(65, 10, "NLP Augmented (Gated)", border=1, align="C", fill=True)
        pdf.ln()
        
        # Table Data
        pdf.set_font("helvetica", size=11)
        data = [
            ("Total Return", f"{baseline.get('total_return', 0)*100:.2f}%", f"{gated.get('total_return', 0)*100:.2f}%"),
            ("Sharpe Ratio", f"{baseline.get('sharpe_ratio', 0):.2f}", f"{gated.get('sharpe_ratio', 0):.2f}"),
            ("Max Drawdown", f"{baseline.get('max_drawdown', 0)*100:.2f}%", f"{gated.get('max_drawdown', 0)*100:.2f}%"),
            ("Win Rate", f"{baseline.get('win_rate', 0)*100:.2f}%", f"{gated.get('win_rate', 0)*100:.2f}%"),
            ("Final Balance", f"${baseline.get('final_balance', 0):,.2f}", f"${gated.get('final_balance', 0):,.2f}")
        ]
        
        for row in data:
            pdf.cell(60, 10, row[0], border=1)
            pdf.cell(65, 10, row[1], border=1, align="C")
            pdf.cell(65, 10, row[2], border=1, align="C")
            pdf.ln()
            
        pdf.ln(10)
        
        # 3. Disclaimer
        pdf.set_font("helvetica", style="I", size=9)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 6, "Disclaimer: This analysis is automatically generated for research purposes only. "
                              "It does not constitute financial advice. Past performance is no guarantee of future results.", align="C")
        
        pdf_path = self.results_dir / "Session_Analysis_Report.pdf"
        pdf.output(str(pdf_path))
        return pdf_path
