import requests
from config import NEWSAPI_KEY, NEWSAPI_URL, TICKERS, START, END, NEWS

def fetch_news(query, from_date=START, to_date=END):
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': NEWSAPI_KEY
    }
    response = requests.get(NEWSAPI_URL, params=params)
    return response.json()

def fetch_gdelt(query):
    params = {'query': query, 'mode': 'ArtList', 'format': 'JSON'}
    response = requests.get(GDELT_URL, params=params)
    return response.json()

import snscrape.modules.twitter as sntwitter
import pandas as pd
from config import TWEET_QUERY_TEMPLATE, START, END, TWEETS

def fetch_tweets(keywords, start=START, end=END, limit=500):
    query = TWEET_QUERY_TEMPLATE.format(
        keywords=" OR ".join(keywords),
        start=start,
        end=end
    )
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        tweets.append([tweet.date, tweet.user.username, tweet.rawContent])
    return pd.DataFrame(tweets, columns=["date", "user", "content"])

# Example
news_data = fetch_news("TotalEnergies")
tweets_df = fetch_tweets(["TotalEnergies", "TTE.PA"])
tweets_df.to_csv(TWEETS / "totalenergies_tweets.csv", index=False)