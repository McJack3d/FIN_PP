"""
Stock price prediction model - Main orchestration module
"""
import os
import json
import warnings
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np

# Import components from other modules
from .features import build_features, log_forward_return, download_data
from .models import create_model, create_ensemble, create_stacked_model
from .backtest import run_backtest
from .evaluation import evaluate_models

try:
    from ..interfaces.utils.results_formatter import ResultsFormatter
except ImportError:
    try:
        from interfaces.utils.results_formatter import ResultsFormatter
    except ImportError:
        # Define a simple fallback formatter
        class ResultsFormatter:
            @staticmethod
            def format_cli(prediction):
                return json.dumps(prediction, indent=2)
            
            @staticmethod
            def format_json(prediction):
                return prediction

def predict_stock(ticker: str = "ALO.PA", years: float = 5.0, horizon: int = 5, threshold: float = 0.60, 
               deadband: float = 0.10, cost_bps: float = 10.0, model_type: str = 'ensemble', 
               calibrate: bool = True, long_only: bool = True, regime_ma: int = 100, 
               vol_target: float = 0.01, outdir: str | None = None, plot: bool = False, 
               random_state: int = 42):
    """
    Main prediction function that orchestrates the entire prediction pipeline
    """
    # Calculate dates
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=int(years * 365))
    
    # Download and process data
    data = download_data(ticker, start_date, end_date)
    if data is None or len(data) < 200:
        raise RuntimeError(f"Not enough data for {ticker}. Got {0 if data is None else len(data)} rows.")
    
    # Build features
    data, features = build_features(data, start_date, end_date)
    
    # Prepare target and features
    y_ret = log_forward_return(data['AdjClose'], horizon).dropna()
    X = data.loc[y_ret.index, features]
    context = data.loc[y_ret.index, ['AdjClose', 'Ret1']].copy()
    
    # Select model and run evaluation
    model_results = evaluate_models(X, y_ret, model_type, calibrate, random_state)
    
    # Run backtest
    backtest_results = run_backtest(
        X, y_ret, context, model_type, threshold, deadband, 
        cost_bps, calibrate, long_only, regime_ma, vol_target, 
        horizon, random_state
    )
    
    # Generate prediction
    prediction = generate_prediction(
        ticker, data, X, y_ret, model_results, 
        backtest_results, horizon, threshold, deadband
    )
    
    # Handle output
    if outdir:
        save_results(prediction, data, outdir)
    
    # Generate visualization
    if plot:
        create_visualization(data, prediction, horizon, outdir)
        
    return prediction

def generate_prediction(ticker, data, X, y_ret, model_results, backtest_results, horizon, threshold, deadband):
    """Generate the final prediction output"""
    # Extract model components
    reg_model = model_results['reg_model']
    clf = model_results['clf_model']
    metrics = model_results['metrics']
    top_features = model_results['top_features']
    
    # Latest features for prediction
    latest_features = X.iloc[[-1]][top_features]
    last_close = float(data.loc[X.index[-1], 'AdjClose'])
    
    # Make predictions
    pred_ret_latest = float(reg_model.predict(latest_features)[0])
    proba_up = float(getattr(clf, 'predict_proba')(latest_features)[0, 1]) if hasattr(clf, 'predict_proba') else 0.5
    
    # Adaptive thresholding
    latest_vol = data['Ret1'].tail(20).std()
    vol_z = (latest_vol - data['Ret1'].tail(100).std()) / data['Ret1'].tail(100).std()
    adaptive_threshold = _adaptive_threshold(vol_z, threshold)
    
    # Decision rule
    if abs(proba_up - 0.5) < deadband:
        decision = 'NO_TRADE'
    else:
        decision = 'UP' if proba_up >= adaptive_threshold else 'DOWN'
    
    # Calculate predicted price
    predicted_price = float(last_close * np.exp(pred_ret_latest))
    
    # Prepare payload
    payload = {
        'ticker': ticker,
        'asof': date.today().isoformat(),
        'horizon_days': horizon,
        'last_close': last_close,
        'predicted_return': pred_ret_latest,
        'predicted_price': predicted_price,
        'y_pred': decision,
        'proba_up': proba_up,
        'threshold': threshold,
        'adaptive_threshold': float(adaptive_threshold),
        'deadband': deadband,
        'metrics': metrics,
        'top_features': top_features,
        # Include backtest results
        'backtest': backtest_results
    }
    
    return payload

def _adaptive_threshold(vol: float, base_threshold: float, vol_scaling: bool = True) -> float:
    """Adjust threshold based on volatility - higher threshold in high volatility periods"""
    if not vol_scaling or np.isnan(vol):
        return base_threshold
    
    # Scale threshold between base and 0.65 based on volatility
    adjusted = base_threshold + min(0.08, max(0, vol * 0.15))
    return min(0.75, adjusted)  # Cap at 0.75

def save_results(prediction, data, outdir):
    """Save prediction results to files"""
    os.makedirs(outdir, exist_ok=True)
    
    # Save raw prediction
    with open(os.path.join(outdir, "latest_prediction.json"), "w") as f:
        json.dump(prediction, f, indent=2)
    
    # Save enhanced results
    enhanced_payload = ResultsFormatter.format_json(prediction)
    with open(os.path.join(outdir, "latest_results.json"), "w") as f:
        json.dump(enhanced_payload, f, indent=2)
    
    # Save simple CSV snapshot
    snap = data.tail(1).copy()
    snap["Predicted_Return"] = prediction['predicted_return']
    snap["Predicted_Price"] = prediction['predicted_price']
    snap["Decision"] = prediction['y_pred']
    snap.to_csv(os.path.join(outdir, "latest_prediction.csv"))

def create_visualization(data, prediction, horizon, outdir=None):
    """Create visualization of the prediction"""
    try:
        import matplotlib.pyplot as plt
        # Create a comprehensive visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Extract prediction details
        last_close = prediction['last_close']
        predicted_price = prediction['predicted_price']
        
        # Price chart with indicators
        tail = data.tail(180)
        ax1.plot(tail.index, tail['AdjClose'], label='Adj Close', linewidth=2)
        ax1.plot(tail.index, tail['MA10'], label='MA10', linestyle='--', alpha=0.7)
        ax1.plot(tail.index, tail['MA30'], label='MA30', linestyle='--', alpha=0.7)
        ax1.plot(tail.index, tail['BBU'], label='BB Upper', linestyle=':', color='gray', alpha=0.6)
        ax1.plot(tail.index, tail['BBL'], label='BB Lower', linestyle=':', color='gray', alpha=0.6)
        ax1.fill_between(tail.index, tail['BBL'], tail['BBU'], color='gray', alpha=0.1)
        
        # Add prediction point
        last_date = tail.index[-1]
        next_date = last_date + pd.Timedelta(days=horizon)
        ax1.plot([last_date, next_date], [last_close, predicted_price], 
               'ro-', linewidth=2, markersize=8, label=f'Prediction ({horizon} days)')
        
        # Add annotations
        ax1.annotate(f'€{last_close:.2f}', (last_date, last_close), 
                    xytext=(10, 0), textcoords='offset points', fontsize=10)
        ax1.annotate(f'€{predicted_price:.2f}', (next_date, predicted_price), 
                    xytext=(10, 0), textcoords='offset points', fontsize=10, 
                    color='darkred', fontweight='bold')
        
        # Styling
        ax1.set_title(f"{prediction['ticker']} Price Prediction - Next {horizon} Days", fontsize=16)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Volume subplot
        ax2.bar(tail.index, tail['Volume'], color='navy', alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if outdir specified
        if outdir:
            plt.savefig(os.path.join(outdir, f"{prediction['ticker']}_prediction.png"), dpi=100, bbox_inches='tight')
        
        plt.show()
    except Exception as e:
        warnings.warn(f"Plotting failed: {e}")
