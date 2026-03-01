import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.absolute()))

from backend.analysis_engine import AnalysisEngine
from backend.backtest_engine import BacktestEngine
import pandas as pd

def test_pdf():
    try:
        ae = AnalysisEngine(Path(".").absolute())
        be = BacktestEngine(Path(".").absolute())
        
        run = be.get_latest_run()
        print(f"Run data: {bool(run)}")
        
        bench_path = Path("benchmark_results.csv")
        bench = []
        if bench_path.exists():
            bench = pd.read_csv(bench_path).to_dict("records")
            
        print(f"Bench data: {len(bench)}")
        
        print("Calling generate_pdf_report...")
        path = ae.generate_pdf_report(run, benchmark_data=bench)
        print(f"Success! {path}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf()
