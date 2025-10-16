from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW = DATA_DIR / "raw"
PRO = DATA_DIR / "processed"

TICKERS = ["TTE.PA", "XOM", "BP", "CVX", "SHEL"]
START, END = "2020-01-01", "2025-01-01"

#--- EQUITY SELECTION ---
NEWSAPI_KEY = "017c2c6a-4bf9-43a6-a0d5-ba4ed68bfb50"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# For reference: params like query=ENERGY or stock symbol, mode=ArtList, format=JSON

# === Twitter (snscrape) ===
TWEET_QUERY_TEMPLATE = '({keywords}) lang:en since:{start} until:{end}'
# e.g. '("TotalEnergies" OR "TTE.PA") lang:en since:2024-01-01 until:2024-12-31'

# === Ensure directories exist ===
RAW.mkdir(parents=True, exist_ok=True)
PRO.mkdir(parents=True, exist_ok=True)
NEWS.mkdir(parents=True, exist_ok=True)
TWEETS.mkdir(parents=True, exist_ok=True)