#!/bin/bash
# Quick start script for Data Engineering Pipeline

echo "=========================================="
echo "Data Engineering Pipeline - Quick Start"
echo "=========================================="
echo ""

# Change to the correct directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Run pipeline
echo ""
echo "=========================================="
echo "Starting Data Engineering Pipeline"
echo "=========================================="
echo ""

cd data
python pipeline.py --mode full

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Processed data is available in:"
echo "  - data/processed/"
echo "  - data/news/"
echo ""
echo "To run again:"
echo "  source venv/bin/activate"
echo "  cd data"
echo "  python pipeline.py --mode full"
