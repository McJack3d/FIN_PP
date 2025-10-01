"""
Command-line interface for stock price prediction
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from the new module structure
from models.stock_predictor import predict_stock

# Import formatter if available
try:
    from interfaces.utils.results_formatter import ResultsFormatter
except ImportError:
    # Use a simple formatter if the module is not available
    class ResultsFormatter:
        @staticmethod
        def format_cli(prediction):
            import json
            return json.dumps(prediction, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Stock price predictor with historical data analysis")
    parser.add_argument("--ticker", default="ALO.PA", help="Ticker (default: ALO.PA)")
    parser.add_argument("--years", type=float, default=5.0, help="Lookback window in years (default: 5)")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.57, help="UP decision threshold (default: 0.57)")
    parser.add_argument("--deadband", type=float, default=0.05, help="Neutral zone around 0.5 (default: 0.05)")
    parser.add_argument("--cost_bps", type=float, default=10.0, help="Per-trade cost in bps (default: 10)")
    parser.add_argument("--model", choices=["histgb", "rf", "stacked", "ensemble"], default="ensemble", 
                        help="Model type (default: ensemble)")
    parser.add_argument("--calibrate", action="store_true", default=True, 
                        help="Calibrate classifier probabilities")
    parser.add_argument("--long_only", action="store_true", default=True, 
                        help="Only take long signals (suppress shorts)")
    parser.add_argument("--regime_ma", type=int, default=100, help="MA window for regime filter (default: 100)")
    parser.add_argument("--vol_target", type=float, default=0.01, 
                        help="Daily volatility target for position sizing (default: 0.01)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for artifacts")
    parser.add_argument("--plot", action="store_true", help="Show visualization plot")

    args = parser.parse_args()
    
    # Run prediction with arguments
    payload = predict_stock(
        ticker=args.ticker,
        years=args.years,
        horizon=args.horizon,
        threshold=args.threshold,
        deadband=args.deadband,
        cost_bps=args.cost_bps,
        model_type=args.model,
        calibrate=args.calibrate,
        long_only=args.long_only,
        regime_ma=args.regime_ma,
        vol_target=args.vol_target,
        outdir=args.outdir,
        plot=args.plot,
    )
    
    # Format and print results
    formatted_output = ResultsFormatter.format_cli(payload)
    print(formatted_output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
