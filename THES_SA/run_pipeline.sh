#!/bin/bash
# THES_SA - Full Pipeline Runner (v2.0)
# Runs all phases: 0 (Audit) -> 1 (Data) -> 2 (Sentiment) -> 3 (Modeling)

echo "=========================================="
echo "THES_SA Pipeline v2.0"
echo "Sentiment-Driven Nuclear Equity Forecasting"
echo "=========================================="
echo ""

# Change to project root
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Dependencies installed"

# Parse arguments
MODE="${1:-full}"
echo ""
echo "Mode: $MODE"
echo ""

case $MODE in
    audit)
        echo "=========================================="
        echo "Phase 0: Feasibility Audit"
        echo "=========================================="
        cd data
        python feasibility_audit.py
        ;;
    collect)
        echo "=========================================="
        echo "Phase 1: Data Collection + Preprocessing"
        echo "=========================================="
        cd data
        python pipeline.py --mode full --skip-audit
        ;;
    sentiment)
        echo "=========================================="
        echo "Phase 2: Sentiment Scoring + Features"
        echo "=========================================="
        cd sentiment
        python scorer.py
        python features.py
        ;;
    model)
        echo "=========================================="
        echo "Phase 3: LSTM Modeling + Evaluation"
        echo "=========================================="
        cd models
        python lstm_model.py
        ;;
    full)
        echo "=========================================="
        echo "Running Full Pipeline (All Phases)"
        echo "=========================================="
        python run_all.py
        ;;
    *)
        echo "Usage: ./run_pipeline.sh [audit|collect|sentiment|model|full]"
        echo ""
        echo "  audit     - Phase 0: FNSPID feasibility audit"
        echo "  collect   - Phase 1: Data collection + preprocessing"
        echo "  sentiment - Phase 2: FinBERT scoring + feature engineering"
        echo "  model     - Phase 3: LSTM training + evaluation + SHAP"
        echo "  full      - All phases (default)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results available in: results/"
echo "Processed data in:    data/processed/"
echo "Trained models in:    models/"
