#!/usr/bin/env python3
"""
GUI entry point for stock price prediction web application
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Start the web application server"""
    print("=" * 70)
    print(f"Starting Stock Price Prediction Web App")
    print(f"Access the web interface at: http://127.0.0.1:8000")
    print("=" * 70)
    
    # Import the app
    try:
        from interfaces.gui.app_launcher import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("\nMake sure all dependencies are installed:")
        print("pip install fastapi uvicorn pandas numpy scikit-learn yfinance matplotlib")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
