from pathlib import Path
import datetime

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW = DATA_DIR / "raw"
PRO = DATA_DIR / "processed"
NEWS = DATA_DIR / "news"
TWEETS = DATA_DIR / "tweets"

TICKERS = ["TTE.PA", "XOM", "BP", "CVX", "SHEL"]
START, END = "2020-01-01", "2025-01-01"

# Replace this with your valid key from newsapi.org
NEWSAPI_KEY = "372adf95863c40a9a9e2f43d65cf84cd"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

TWEET_QUERY_TEMPLATE = '({keywords}) lang:en since:{start} until:{end}'

# === Ensure directories exist ===
RAW.mkdir(parents=True, exist_ok=True)
PRO.mkdir(parents=True, exist_ok=True)
NEWS.mkdir(parents=True, exist_ok=True)
TWEETS.mkdir(parents=True, exist_ok=True)