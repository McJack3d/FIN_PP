#!/usr/bin/env python3
"""
Command-line entry point for stock price prediction
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the CLI module
from interfaces.cli.stock_predictor_cli import main

if __name__ == "__main__":
    sys.exit(main())
