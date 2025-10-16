import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config import PRO, RAW
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_model_data(ticker, target_col='daily_change', forecast_horizon=1, 
                      features=None, include_sentiment=True, include_technical=True):
    """
    Prepare data for model training
    
    Args:
        ticker (str): Ticker symbol
        target_col (str): Target column for prediction
        forecast_horizon (int): Days ahead to forecast
        features (list): List of feature columns to use
        include_sentiment (bool): Whether to include sentiment features
        include_technical (bool): Whether to include technical indicators
    
    Returns:
        tuple: X (features), y (target), feature_names
    """
    clean_ticker = ticker.replace('.', '_')
    
    # Path to merged price and sentiment data
    data_path = PRO / f"{clean_ticker}_prices_sentiment.csv"
    
    if not data_path.exists():
        logger.error(f"Merged data file not found for {ticker}")
        return None, None, None
    
    logger.info(f"Preparing model data for {ticker}")
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # Create shifted target variable (future return)
    if target_col in df.columns:
        df[f'target_{forecast_horizon}d'] = df[target_col].shift(-forecast_horizon)
    else:
        logger.error(f"Target column {target_col} not found in data")
        return None, None, None
    
    # Create feature list based on parameters
    if features is None:
        features = []
        
        # Add sentiment features if available and requested
        if include_sentiment:
            sentiment_features = ['sentiment_mean', 'sentiment_std', 'article_count',
                               'title_sentiment_mean', 'finbert_mean', 'finbert_std']
            # Only include features that exist in the dataframe
            sentiment_features = [f for f in sentiment_features if f in df.columns]
            features.extend(sentiment_features)
        
        # Add technical indicators if requested
        if include_technical:
            tech_features = ['ma_5', 'ma_20', 'ma_50', 'volatility']
            tech_features = [f for f in tech_features if f in df.columns]
            features.extend(tech_features)
            
        # Add other potential features
        other_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        other_features = [f for f in other_features if f in df.columns]
        features.extend(other_features)
    
    # Create feature matrix and target vector
    X = df[features].copy()
    y = df[f'target_{forecast_horizon}d'].copy()
    
    # Remove rows with NaN values
    valid_rows = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_rows]
    y = y[valid_rows]
    
    return X, y, features

def train_model(X, y, model_type='rf', test_size=0.2, random_state=42):
    """
    Train a machine learning model
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target vector
        model_type (str): Model type ('rf', 'gb', 'lr', 'xgb')
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed
    
    Returns:
        tuple: model, X_train, X_test, y_train, y_test, scaler
    """
    if X is None or y is None:
        logger.error("Cannot train model: input data is None")
        return None, None, None, None, None, None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    logger.info(f"Training {model_type.upper()} model with {X_train.shape[1]} features")
    
    # Initialize model based on type
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    elif model_type == 'lr':
        model = LinearRegression()
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None, None, None, None, None, None
    
    # Train model
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test (DataFrame): Test features
        y_test (Series): Test target
        feature_names (list): Feature names
    
    Returns:
        dict: Evaluation metrics
    """
    if model is None or X_test is None or y_test is None:
        logger.error("Cannot evaluate model: input data is None")
        return None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    logger.info(f"Model evaluation metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_importance = {}
        for i in range(len(indices)):
            feature_importance[feature_names[indices[i]]] = importances[indices[i]]
        
        metrics['feature_importance'] = feature_importance
    
    return metrics

def save_model(model, scaler, ticker, model_type, metrics, features):
    """
    Save trained model and metadata
    
    Args:
        model: Trained model
        scaler: Feature scaler
        ticker (str): Ticker symbol
        model_type (str): Model type
        metrics (dict): Evaluation metrics
        features (list): Feature names
    
    Returns:
        str: Path to saved model
    """
    # Create models directory if it doesn't exist
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Clean ticker name for filename
    clean_ticker = ticker.replace('.', '_')
    
    # Create filename with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    filename = f"{clean_ticker}_{model_type}_model_{timestamp}.joblib"
    filepath = models_dir / filename
    
    # Save model, scaler and metadata
    model_data = {
        'model': model,
        'scaler': scaler,
        'ticker': ticker,
        'model_type': model_type,
        'metrics': metrics,
        'features': features,
        'timestamp': timestamp
    }
    
    joblib.dump(model_data, filepath)
    logger.info(f"Model saved to {filepath}")
    
    return str(filepath)

def train_and_evaluate(ticker, model_type='rf', forecast_horizon=1):
    """
    End-to-end function to train and evaluate a model for a ticker
    
    Args:
        ticker (str): Ticker symbol
        model_type (str): Model type ('rf', 'gb', 'lr', 'xgb')
        forecast_horizon (int): Days ahead to forecast
    
    Returns:
        dict: Results including model, metrics, and paths
    """
    # Prepare data
    X, y, features = prepare_model_data(ticker, forecast_horizon=forecast_horizon)
    
    if X is None or y is None:
        logger.error(f"Failed to prepare data for {ticker}")
        return None
    
    # Train model
    model, X_train, X_test, y_train, y_test, scaler = train_model(
        X, y, model_type=model_type
    )
    
    if model is None:
        logger.error(f"Failed to train model for {ticker}")
        return None
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, features)
    
    if metrics is None:
        logger.error(f"Failed to evaluate model for {ticker}")
        return None
    
    # Save model
    model_path = save_model(model, scaler, ticker, model_type, metrics, features)
    
    return {
        'model': model,
        'scaler': scaler,
        'features': features,
        'metrics': metrics,
        'model_path': model_path
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models for stock prediction")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to model")
    parser.add_argument("--model", default="rf", choices=["rf", "gb", "lr", "xgb"], 
                       help="Model type (rf=RandomForest, gb=GradientBoosting, lr=LinearRegression, xgb=XGBoost)")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in days")
    
    args = parser.parse_args()
    
    # Train and evaluate model
    results = train_and_evaluate(args.ticker, args.model, args.horizon)
    
    if results:
        logger.info(f"Successfully trained and evaluated {args.model} model for {args.ticker}")
        logger.info(f"R² score: {results['metrics']['r2']:.4f}")
    else:
        logger.error(f"Failed to train and evaluate model for {args.ticker}")