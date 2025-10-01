"""
Launcher script for the Stock Prediction Web App
"""
import os
import sys
from pathlib import Path

# Add the project root directory to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can safely import our app
from webapi.main import app

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print(f"Starting Stock Prediction Web App")
    print(f"Access the web interface at: http://127.0.0.1:8000")
    print("=" * 70)
    
    uvicorn.run(
        "webapi.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=[str(project_root)]
    )
