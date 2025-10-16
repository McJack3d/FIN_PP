import sys
from pathlib import Path
import os

def test_imports():
    """Test if all required packages are available"""
    packages = {
        "requests": "requests",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "sklearn": "scikit-learn",
        "yfinance": "yfinance",
        "plotly": "plotly"
    }
    
    optional_packages = {
        "snscrape": "snscrape",
        "torch": "torch",
        "transformers": "transformers"
    }
    
    print("=== Testing Required Packages ===")
    for module_name, pip_name in packages.items():
        try:
            __import__(module_name)
            print(f"✓ {module_name} is available")
        except ImportError:
            print(f"✗ {module_name} is missing - run: pip install {pip_name}")
    
    print("\n=== Testing Optional Packages ===")
    for module_name, pip_name in optional_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {module_name} is available")
        except ImportError:
            print(f"✗ {module_name} is missing - run: pip install {pip_name}")


def test_project_structure():
    """Test project folder structure"""
    print("\n=== Testing Project Structure ===")
    
    # Check config import
    try:
        sys.path.append(os.path.dirname(__file__))
        from config import DATA_DIR, RAW, PRO, NEWS, TWEETS
        print("✓ config.py imported successfully")
        
        # Check directory existence
        for name, path in [("DATA_DIR", DATA_DIR), ("RAW", RAW), 
                          ("PRO", PRO), ("NEWS", NEWS), ("TWEETS", TWEETS)]:
            if path.exists():
                print(f"✓ {name} directory exists at {path}")
            else:
                print(f"✗ {name} directory missing at {path}")
    except Exception as e:
        print(f"✗ Failed to import config: {str(e)}")


def test_api_connections():
    """Test API connections"""
    print("\n=== Testing API Connections ===")
    
    # Test NewsAPI
    try:
        from config import NEWSAPI_KEY
        import requests
        
        if NEWSAPI_KEY == "017c2c6a-4bf9-43a6-a0d5-ba4ed68bfb50":
            print("! Using default API key - you may need to register for a valid key")
        
        test_url = "https://newsapi.org/v2/top-headlines"
        params = {
            'country': 'us',
            'category': 'business',
            'apiKey': NEWSAPI_KEY
        }
        
        response = requests.get(test_url, params=params)
        if response.status_code == 200:
            print(f"✓ NewsAPI connection successful - status code {response.status_code}")
        else:
            print(f"✗ NewsAPI connection failed - status code {response.status_code}")
            print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"✗ NewsAPI test failed: {str(e)}")
    
    # Test GDELT connection
    try:
        from config import GDELT_URL
        import requests
        
        params = {'query': 'energy', 'mode': 'ArtList', 'format': 'JSON', 'maxrecords': 1}
        response = requests.get(GDELT_URL, params=params)
        
        if response.status_code == 200:
            print(f"✓ GDELT API connection successful - status code {response.status_code}")
        else:
            print(f"✗ GDELT API connection failed - status code {response.status_code}")
    except Exception as e:
        print(f"✗ GDELT API test failed: {str(e)}")


def test_fetch_module():
    """Test fetch module functionality"""
    print("\n=== Testing Fetch Module ===")
    
    try:
        import fetch
        print("✓ fetch.py imported successfully")
        
        # Test fetch_news function with minimal request
        try:
            test_result = fetch.fetch_news("energy", "2023-01-01", "2023-01-02")
            if 'articles' in test_result:
                print(f"✓ fetch_news function works - found {len(test_result['articles'])} articles")
            else:
                print(f"✗ fetch_news returned unexpected response: {test_result}")
        except Exception as e:
            print(f"✗ fetch_news test failed: {str(e)}")
        
        # Note about fetch_tweets
        print("! fetch_tweets not tested - requires snscrape and can be rate-limited")
        
    except Exception as e:
        print(f"✗ Failed to import fetch module: {str(e)}")


def test_yfinance():
    """Test yfinance functionality"""
    print("\n=== Testing yfinance Module ===")
    
    try:
        import yfinance as yf
        
        # Test with a single ticker
        test_ticker = "XOM"
        print(f"Fetching recent data for {test_ticker}...")
        
        ticker_data = yf.download(test_ticker, period="1mo", progress=False)
        if not ticker_data.empty:
            print(f"✓ yfinance works - fetched {len(ticker_data)} rows for {test_ticker}")
            print(f"  Date range: {ticker_data.index.min().date()} to {ticker_data.index.max().date()}")
        else:
            print(f"✗ No data returned for {test_ticker}")
        
    except Exception as e:
        print(f"✗ yfinance test failed: {str(e)}")


if __name__ == "__main__":
    print("===== Testing Project Setup =====")
    print("Testing environment and dependencies for THES_SA project")
    print("=============================================")
    
    # Run all tests
    test_imports()
    test_project_structure()
    test_api_connections()
    test_fetch_module()
    test_yfinance()
    
    print("\n===== Setup Tests Complete =====")