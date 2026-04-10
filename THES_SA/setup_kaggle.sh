#!/usr/bin/env bash
# ============================================================
# THES_SA Kaggle Setup Script
# Uploads code as a Kaggle dataset and pushes the kernel.
#
# Prerequisites:
#   1. Install Kaggle CLI: pip install kaggle
#   2. Configure API key: ~/.kaggle/kaggle.json
#      (Download from https://www.kaggle.com/settings -> API -> Create New Token)
#
# Usage:
#   cd THES_SA
#   bash setup_kaggle.sh              # Upload dataset + push kernel
#   bash setup_kaggle.sh --dataset    # Upload/update code dataset only
#   bash setup_kaggle.sh --kernel     # Push kernel only
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ---- Check prerequisites ----
command -v kaggle >/dev/null 2>&1 || error "Kaggle CLI not found. Install with: pip install kaggle"

if [ ! -f ~/.kaggle/kaggle.json ]; then
    error "Kaggle API key not found at ~/.kaggle/kaggle.json
    Download from: https://www.kaggle.com/settings -> API -> Create New Token"
fi

# ---- Parse arguments ----
DO_DATASET=true
DO_KERNEL=true

if [ "${1:-}" = "--dataset" ]; then
    DO_KERNEL=false
elif [ "${1:-}" = "--kernel" ]; then
    DO_DATASET=false
fi

# ---- Step 1: Upload code as a Kaggle dataset ----
if [ "$DO_DATASET" = true ]; then
    info "Uploading THES_SA code as Kaggle dataset..."

    # dataset-metadata.json must be in the directory being uploaded
    if [ ! -f dataset-metadata.json ]; then
        error "dataset-metadata.json not found in $SCRIPT_DIR"
    fi

    # Create or update dataset
    if kaggle datasets status alexandrebredillot/thes-sa-code 2>/dev/null | grep -q "ready"; then
        info "Dataset exists, creating new version..."
        kaggle datasets version -p . -m "Updated $(date +%Y-%m-%d)" -r zip -q
    else
        info "Creating new dataset..."
        kaggle datasets create -p . -r zip -q
    fi

    info "Code dataset uploaded successfully."
    info "View at: https://www.kaggle.com/datasets/alexandrebredillot/thes-sa-code"
    echo ""
fi

# ---- Step 2: Push kernel ----
if [ "$DO_KERNEL" = true ]; then
    info "Pushing Kaggle kernel..."

    if [ ! -f kernel-metadata.json ]; then
        error "kernel-metadata.json not found in $SCRIPT_DIR"
    fi

    kaggle kernels push -p .

    info "Kernel pushed successfully."
    info "View at: https://www.kaggle.com/code/alexandrebredillot/thes-sa-sentiment-nuclear-forecasting"
    echo ""
fi

# ---- Summary ----
info "============================================================"
info "Setup complete!"
info ""
info "Attached datasets:"
info "  1. FNSPID: elsabetyemane/financial-news-and-stock-price-integration-dataset"
info "  2. Code:   alexandrebredillot/thes-sa-code"
info ""
info "Kernel settings:"
info "  GPU:      enabled"
info "  Internet: enabled"
info ""
info "To check kernel status:"
info "  kaggle kernels status alexandrebredillot/thes-sa-sentiment-nuclear-forecasting"
info ""
info "To pull output after completion:"
info "  kaggle kernels output alexandrebredillot/thes-sa-sentiment-nuclear-forecasting -p ./kaggle_output"
info "============================================================"
