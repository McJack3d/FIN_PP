from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW = DATA_DIR / "raw"
PRO = DATA_DIR / "processed"

TICKERS = ["TTE.PA", "XOM", "BP", "CVX", "SHEL"]
START, END = "2020-01-01", "2025-01-01"

RAW.mkdir(parents=True, exist_ok=True)
PRO.mkdir(parents=True, exist_ok=True)