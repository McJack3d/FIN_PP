"""
Results formatter for the stock prediction model.
This module transforms raw model outputs into more user-friendly and actionable formats.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

class ResultsFormatter:
    """Format model results for different output formats"""
    
    @staticmethod
    def format_json(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Format the prediction result as enhanced JSON"""
        # Extract key components
        ticker = prediction['ticker']
        horizon = prediction['horizon_days']
        last_close = prediction['last_close']
        predicted_price = prediction['predicted_price']
        decision = prediction['y_pred']
        
        # Calculate derived metrics
        expected_return = (predicted_price / last_close - 1) * 100
        confidence = prediction['proba_up'] if decision == 'UP' else (1 - prediction['proba_up'])
        
        # Create target date
        target_date = (datetime.now() + timedelta(days=horizon)).strftime('%Y-%m-%d')
        
        # Format metrics for display
        metrics = prediction['metrics']
        backtest = metrics['backtest']
        
        # Generate actionable recommendation
        recommendation = ResultsFormatter._generate_recommendation(prediction)
        
        # Risk assessment
        risk_assessment = ResultsFormatter._assess_risk(prediction)
        
        # Return enhanced result
        return {
            "ticker": ticker,
            "analysis_date": datetime.now().strftime('%Y-%m-%d'),
            "forecast": {
                "direction": decision,
                "confidence": round(confidence * 100, 1),
                "last_price": round(last_close, 2),
                "target_price": round(predicted_price, 2),
                "expected_return": round(expected_return, 2),
                "horizon_days": horizon,
                "target_date": target_date
            },
            "recommendation": recommendation,
            "risk": risk_assessment,
            "model_performance": {
                "accuracy": round(metrics['clf_accuracy_cv_mean'] * 100, 1),
                "auc_score": round(metrics['clf_auc_cv_mean'], 3),
                "backtest_return": round(backtest['cum_return'] * 100, 1),
                "sharpe_ratio": round(backtest['sharpe_annual'], 2),
                "hit_ratio": round(backtest['hit_ratio'] * 100, 1),
                "max_drawdown": round(backtest['max_drawdown'] * 100, 1),
                "trades": backtest['trades']
            },
            "key_drivers": prediction.get('top_features', [])[:5]
        }
    
    @staticmethod
    def format_cli(prediction: Dict[str, Any]) -> str:
        """Format the prediction result for CLI display"""
        # Get enhanced JSON first
        enhanced = ResultsFormatter.format_json(prediction)
        
        # Build a visually appealing CLI output
        forecast = enhanced['forecast']
        model = enhanced['model_performance']
        
        # Direction indicator
        if forecast['direction'] == 'UP':
            direction_symbol = "↑"
            direction_text = "BULLISH"
        elif forecast['direction'] == 'DOWN':
            direction_symbol = "↓"
            direction_text = "BEARISH"
        else:
            direction_symbol = "→"
            direction_text = "NEUTRAL"
            
        # Build the output string
        output = [
            f"\n{'=' * 60}",
            f"  STOCK PREDICTION REPORT: {enhanced['ticker']}",
            f"{'=' * 60}",
            f"\n  FORECAST: {direction_symbol} {direction_text} {direction_symbol}  ({forecast['confidence']}% confidence)",
            f"  Horizon: {forecast['horizon_days']} days (until {forecast['target_date']})",
            f"\n  Current Price: €{forecast['last_price']:.2f}",
            f"  Target Price:  €{forecast['target_price']:.2f}",
            f"  Expected Return: {forecast['expected_return']:+.2f}%",
            f"\n  RECOMMENDATION:",
            f"  {enhanced['recommendation']}",
            f"\n  RISK ASSESSMENT:",
            f"  {enhanced['risk']}",
            f"\n  MODEL PERFORMANCE:",
            f"  Accuracy: {model['accuracy']}%   AUC Score: {model['auc_score']}",
            f"  Backtest Return: {model['backtest_return']}%   Sharpe Ratio: {model['sharpe_ratio']}",
            f"  Hit Ratio: {model['hit_ratio']}%   Max Drawdown: {model['max_drawdown']}%",
            f"\n  KEY DRIVERS:",
            f"  {', '.join(enhanced['key_drivers'][:5])}",
            f"\n{'=' * 60}\n"
        ]
        
        return "\n".join(output)
    
    @staticmethod
    def format_html(prediction: Dict[str, Any]) -> str:
        """Format the prediction result as HTML"""
        # Get enhanced JSON first
        enhanced = ResultsFormatter.format_json(prediction)
        forecast = enhanced['forecast']
        
        # Determine CSS classes based on direction
        if forecast['direction'] == 'UP':
            direction_class = "text-success"
            direction_text = "BULLISH"
        elif forecast['direction'] == 'DOWN':
            direction_class = "text-danger"
            direction_text = "BEARISH"
        else:
            direction_class = "text-warning"
            direction_text = "NEUTRAL"
        
        # Build HTML
        html = f"""
        <div class="prediction-result">
            <h3 class="ticker-heading">{enhanced['ticker']} - {direction_text}</h3>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Price Forecast</div>
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <span>Current Price:</span>
                                <span class="fw-bold">€{forecast['last_price']:.2f}</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span>Target Price:</span>
                                <span class="fw-bold {direction_class}">€{forecast['target_price']:.2f}</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span>Expected Return:</span>
                                <span class="fw-bold {direction_class}">{forecast['expected_return']:+.2f}%</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span>Target Date:</span>
                                <span>{forecast['target_date']}</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span>Confidence:</span>
                                <span>{forecast['confidence']}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Recommendation</div>
                        <div class="card-body">
                            <p>{enhanced['recommendation']}</p>
                            <p><small class="text-muted">Risk: {enhanced['risk']}</small></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        return html
    
    @staticmethod
    def _generate_recommendation(prediction: Dict[str, Any]) -> str:
        """Generate an actionable recommendation based on the prediction"""
        decision = prediction['y_pred']
        confidence = prediction['proba_up'] if decision == 'UP' else (1 - prediction['proba_up'])
        confidence_pct = confidence * 100
        ticker = prediction['ticker']
        horizon = prediction['horizon_days']
        metrics = prediction['metrics']
        backtest = metrics['backtest']
        
        # Base recommendation on direction and confidence
        if decision == 'UP' and confidence_pct >= 65:
            action = f"STRONG BUY - Consider building a position in {ticker}."
        elif decision == 'UP' and confidence_pct >= 55:
            action = f"BUY - Consider opportunistic purchases of {ticker}."
        elif decision == 'UP':
            action = f"WEAK BUY - Monitor {ticker} for better entry points."
        elif decision == 'DOWN' and confidence_pct >= 65:
            action = f"STRONG SELL - Consider reducing exposure to {ticker}."
        elif decision == 'DOWN' and confidence_pct >= 55:
            action = f"SELL - Consider gradual reduction of {ticker} position."
        elif decision == 'DOWN':
            action = f"WEAK SELL - Exercise caution with {ticker}."
        else:
            action = f"NEUTRAL - Hold current positions in {ticker}."
        
        # Consider model performance
        if backtest['hit_ratio'] < 0.5:
            caution = f" Note: The model's historical accuracy is below average."
            action += caution
        
        # Add time horizon
        action += f" The forecast horizon is {horizon} trading days."
        
        return action
    
    @staticmethod
    def _assess_risk(prediction: Dict[str, Any]) -> str:
        """Provide a risk assessment for the prediction"""
        metrics = prediction['metrics']
        backtest = metrics['backtest']
        
        # Assess risk based on max drawdown and volatility
        if backtest['max_drawdown'] < -0.15:
            risk_level = "High risk"
            explanation = "substantial historical drawdowns"
        elif backtest['max_drawdown'] < -0.10:
            risk_level = "Moderate-to-high risk"
            explanation = "significant historical volatility"
        elif backtest['max_drawdown'] < -0.05:
            risk_level = "Moderate risk"
            explanation = "moderate historical volatility"
        else:
            risk_level = "Lower risk"
            explanation = "relatively stable historical performance"
        
        # Adjust based on model confidence
        decision = prediction['y_pred']
        confidence = prediction['proba_up'] if decision == 'UP' else (1 - prediction['proba_up'])
        
        if confidence < 0.55:
            confidence_note = " The model's confidence is relatively low, suggesting higher uncertainty."
        elif confidence < 0.65:
            confidence_note = " The model shows moderate confidence in this prediction."
        else:
            confidence_note = " The model shows high confidence in this prediction."
        
        return f"{risk_level} due to {explanation}.{confidence_note}"


if __name__ == "__main__":
    # Example usage
    from ALST_Ticker_Predictor_Final import run
    
    # Run a prediction
    result = run(ticker="ALO.PA", horizon=5)
    
    # Format as JSON
    json_result = ResultsFormatter.format_json(result)
    print(json.dumps(json_result, indent=2))
    
    # Format for CLI
    cli_output = ResultsFormatter.format_cli(result)
    print(cli_output)
