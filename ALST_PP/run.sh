#!/bin/bash

# Define the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Try different Python commands
if command -v python3 &> /dev/null; then
    echo "Using python3..."
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    echo "Using python..."
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.x"
    exit 1
fi

# Display menu of options
echo "============================================="
echo "Stock Price Prediction Tool"
echo "============================================="
echo "1) Start Web Interface (GUI)"
echo "2) Run CLI Prediction"
echo "3) Run Notebook for Model Debugging"
echo "4) Exit"
echo "============================================="
read -p "Choose an option (1-4): " option

case "$option" in
    1)
        echo "Starting Web Interface..."
        $PYTHON_CMD gui.py
        ;;
    2)
        echo "Running CLI Prediction..."
        read -p "Enter ticker symbol (default: ALO.PA): " ticker
        ticker=${ticker:-ALO.PA}
        read -p "Enter prediction horizon in days (default: 5): " horizon
        horizon=${horizon:-5}
        $PYTHON_CMD cli.py --ticker "$ticker" --horizon "$horizon" --plot
        ;;
    3)
        # Check if we have jupyter installed
        if command -v jupyter &> /dev/null; then
            echo "Starting Jupyter notebook..."
            jupyter notebook interfaces/notebooks/Model_Debugging.ipynb
        else
            echo "Jupyter not found. Please install with:"
            echo "pip install jupyter"
        fi
        ;;
    4)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac
