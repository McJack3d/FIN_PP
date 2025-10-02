"""FastAPI web interface for stock predictor. Launch with: uvicorn main:app --reload"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
import traceback

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from interfaces.cli.stock_predictor_legacy import run

app = FastAPI(title="Stock Price Prediction API")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "template"))

# Try to mount static files if directory exists
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    ticker: str = Form(...),
    years: float = Form(5.0),
    horizon: int = Form(5),
    threshold: float = Form(0.62),
    ensemble: bool = Form(True),
    calibrate: bool = Form(True)
):
    try:
        # Run the prediction model
        result = run(
            ticker=ticker,
            years=years,
            horizon=horizon,
            threshold=threshold,
            calibrate=calibrate,
            use_ensemble=ensemble
        )
        
        # Enhance the result with additional useful information
        enhanced_result = enhance_prediction_result(result)
        
        return JSONResponse({
            "success": True, 
            "ticker": ticker,
            "prediction": enhanced_result,
            "message": f"Prediction completed for {ticker}"
        })
    except Exception as e:
        error_details = str(e)
        
        # In development, include stack trace
        if app.debug:
            error_details += "\n" + traceback.format_exc()
            
        return JSONResponse({
            "success": False, 
            "error": error_details
        }, status_code=500)

@app.get("/predict")
async def predict_get(
    ticker: str = "ALO.PA",
    years: float = 5.0,
    horizon: int = 5,
    threshold: float = 0.62,
    ensemble: bool = True,
    calibrate: bool = True
):
    """GET endpoint for debugging - redirect to POST or handle directly"""
    try:
        result = run(
            ticker=ticker,
            years=years,
            horizon=horizon,
            threshold=threshold,
            calibrate=calibrate,
            use_ensemble=ensemble
        )
        
        enhanced_result = enhance_prediction_result(result)
        
        return JSONResponse({
            "success": True, 
            "ticker": ticker,
            "prediction": enhanced_result,
            "message": f"Prediction completed for {ticker} (via GET)",
            "note": "Consider using POST method for better performance"
        })
    except Exception as e:
        return JSONResponse({
            "success": False, 
            "error": str(e)
        }, status_code=500)

def enhance_prediction_result(result):
    """Add additional useful information to the prediction result"""
    
    # Calculate additional metrics
    expected_return_pct = (result['predicted_price'] / result['last_close'] - 1) * 100
    
    # Add clearer interpretation of the prediction
    decision_confidence = result['proba_up'] if result['y_pred'] == 'UP' else (1 - result['proba_up'])
    
    # Generate plain language interpretation
    interpretation = generate_interpretation(result)
    
    # Add market context analysis
    market_context = analyze_market_context(result)
    
    # Enhance the result
    result.update({
        'expected_return_pct': float(expected_return_pct),
        'decision_confidence': float(decision_confidence),
        'interpretation': interpretation,
        'market_context': market_context,
        'time_horizon': {
            'days': result['horizon_days'],
            'target_date': (datetime.now() + timedelta(days=result['horizon_days'])).strftime('%Y-%m-%d')
        }
    })
    
    return result

def generate_interpretation(result):
    """Generate a plain language interpretation of the prediction"""
    ticker = result['ticker']
    horizon = result['horizon_days']
    
    if result['y_pred'] == 'UP':
        confidence = result['proba_up'] * 100
        return (
            f"The model predicts {ticker} will rise over the next {horizon} days with "
            f"{confidence:.1f}% confidence. The predicted price target is €{result['predicted_price']:.2f}, "
            f"representing a {(result['predicted_price'] / result['last_close'] - 1) * 100:.1f}% gain from "
            f"the current price of €{result['last_close']:.2f}."
        )
    elif result['y_pred'] == 'DOWN':
        confidence = (1 - result['proba_up']) * 100
        return (
            f"The model predicts {ticker} will decline over the next {horizon} days with "
            f"{confidence:.1f}% confidence. The predicted price target is €{result['predicted_price']:.2f}, "
            f"representing a {(result['predicted_price'] / result['last_close'] - 1) * 100:.1f}% change from "
            f"the current price of €{result['last_close']:.2f}."
        )
    else:
        return (
            f"The model suggests a neutral outlook for {ticker} over the next {horizon} days. "
            f"The predicted price of €{result['predicted_price']:.2f} is not significantly different "
            f"from the current price of €{result['last_close']:.2f}."
        )

def analyze_market_context(result):
    """Add market context analysis to the prediction"""
    # This would ideally pull in more market data, but we'll simulate for now
    metrics = result['metrics']
    backtest = metrics['backtest']
    
    # Compare model performance to buy & hold
    outperformance = backtest['cum_return'] - metrics.get('buyhold_cum_return', 0)
    
    # Analyze hit ratio
    hit_ratio_quality = "excellent" if backtest['hit_ratio'] > 0.6 else \
                        "good" if backtest['hit_ratio'] > 0.55 else \
                        "fair" if backtest['hit_ratio'] > 0.5 else "poor"
    
    # Analyze Sharpe ratio
    sharpe_quality = "excellent" if backtest['sharpe_annual'] > 1.5 else \
                     "good" if backtest['sharpe_annual'] > 1.0 else \
                     "fair" if backtest['sharpe_annual'] > 0.5 else "poor"
    
    return {
        "model_quality": {
            "hit_ratio": backtest['hit_ratio'],
            "hit_ratio_assessment": hit_ratio_quality,
            "sharpe_ratio": backtest['sharpe_annual'],
            "sharpe_assessment": sharpe_quality
        },
        "relative_performance": {
            "outperformance": outperformance,
            "outperformance_pct": outperformance * 100,
            "trades": backtest['trades']
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    app.debug = True
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)