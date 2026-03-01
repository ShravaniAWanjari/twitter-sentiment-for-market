import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from backend.analysis_engine import AnalysisEngine
from backend.backtest_engine import BacktestEngine

def diag():
    ae = AnalysisEngine()
    be = BacktestEngine(Path(__file__).parent)
    
    run = be.get_latest_run()
    if not run:
        print("No run.")
        return
        
    bench = []
    bench_p = Path("benchmark_results.csv")
    if bench_p.exists():
        import pandas as pd
        df = pd.read_csv(bench_p)
        bench = df.to_dict(orient="records")
        
    try:
        pdf_path = ae.generate_pdf_report(run, benchmark_data=bench)
        print(f"Success! {pdf_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diag()
