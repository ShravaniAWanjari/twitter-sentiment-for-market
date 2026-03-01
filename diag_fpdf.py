try:
    from fpdf import FPDF
    print("Successfully imported FPDF")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Other Error: {e}")
