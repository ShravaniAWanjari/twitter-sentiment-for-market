"""Diagnose why IG and Grad*Input return empty results."""
import sys
import io
sys.path.insert(0, r"c:\Users\Lenovo\OneDrive\Desktop\12 week thing\Capstone")

output_lines = []
def log(msg=""):
    output_lines.append(str(msg))
    print(msg)

import torch
import torch.nn.functional as F
from model_factory import load_model

text = "bitcoin is definitely going down"
model_name = "modernbert"

wrapper = load_model(model_name)
wrapper.eval()

inputs = wrapper.encode_texts([text], return_tensors="pt")
input_ids = inputs["input_ids"][0]
attn_mask = inputs.get("attention_mask")

log("=== Base forward ===")
with torch.no_grad():
    out = wrapper(**inputs)
    probs = F.softmax(out.logits, dim=-1).squeeze(0)
    log("probs: " + str([round(x,4) for x in probs.tolist()]))
    log("n_classes: " + str(probs.shape[0]))

log("\n=== Embedding layers ===")
emb_layer = None
for name, mod in wrapper.model.named_modules():
    if "embed" in name.lower() and hasattr(mod, "weight") and len(mod.weight.shape) == 2:
        log("  " + name + ": " + str(mod.weight.shape))
        if mod.weight.shape[0] > 5000 and emb_layer is None:
            emb_layer = mod
            log("  -> Using this one")

log("\n=== inputs_embeds test ===")
if emb_layer is not None:
    with torch.no_grad():
        raw_embeds = emb_layer(input_ids).detach().clone()
    log("raw_embeds shape: " + str(raw_embeds.shape))
    
    leaf = raw_embeds.unsqueeze(0).requires_grad_(True)
    try:
        with torch.enable_grad():
            out2 = wrapper.model(inputs_embeds=leaf, attention_mask=attn_mask)
            bullish_score = F.softmax(out2.logits, dim=-1)[0, 2]
            log("bullish_score: " + str(round(bullish_score.item(), 4)))
            bullish_score.backward()
            log("leaf.grad is None: " + str(leaf.grad is None))
            if leaf.grad is not None:
                gi = (leaf.grad * leaf).squeeze(0).sum(dim=-1).detach()
                log("gi (first 5): " + str([round(x,4) for x in gi[:5].tolist()]))
                log("gi max abs: " + str(round(gi.abs().max().item(), 4)))
    except Exception as e:
        log("inputs_embeds FAILED: " + str(e))
        import traceback
        log(traceback.format_exc())
else:
    log("No emb_layer found!")

log("\n=== Wrapper model type ===")
log("wrapper type: " + type(wrapper).__name__)
log("wrapper.model type: " + type(wrapper.model).__name__)
log("has encode_texts: " + str(hasattr(wrapper, 'encode_texts')))

# Write to file
with open(r"c:\Users\Lenovo\OneDrive\Desktop\12 week thing\Capstone\tmp_diag_out.txt", "w") as f:
    f.write("\n".join(output_lines))
print("Written to tmp_diag_out.txt")
