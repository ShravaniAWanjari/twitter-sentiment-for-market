import sys
import json
sys.path.insert(0, r"c:\Users\Lenovo\OneDrive\Desktop\12 week thing\Capstone")

from backend.analysis_engine import AnalysisEngine
from pathlib import Path

ae = AnalysisEngine(Path(r"c:\Users\Lenovo\OneDrive\Desktop\12 week thing\Capstone"))
res = ae.explain_multi("modernbert", "bitcoin price is falling rapidly")

print("Keys in result:", list(res.keys()))
print("Methods available:", list(res['methods'].keys()))
for m in res['methods']:
    tokens = res['methods'][m]['tokens']
    print(f"Method {m}: {len(tokens)} tokens found")
    if tokens:
        print(f"  First 3 tokens: {tokens[:3]}")

with open(r"c:\Users\Lenovo\OneDrive\Desktop\12 week thing\Capstone\tmp_multi_check.txt", "w") as f:
    f.write(json.dumps(res, indent=2))
