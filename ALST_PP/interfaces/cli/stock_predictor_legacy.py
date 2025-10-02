import argparse
import json
import os
import warnings
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.feature_selection import SelectFromModel

# -----------------------------
# Feature engineering
# -----------------------------

def _compute_obv_from_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = [0.0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + float(volume.iloc[i]))
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - float(volume.iloc[i]))
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)



def _pct_change_safe(s: pd.Series, periods: int = 1) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        r = s.pct_change(periods=periods)
    return r.replace([np.inf, -np.inf], np.nan)


def _log_forward_return(s: pd.Series, horizon: int) -> pd.Series:
    """Compute forward log-return over `horizon` days: ln(P_{t+h}/P_t) aligned at t."""
    ln = np.log(s.astype(float))
    return (ln.shift(-horizon) - ln)


def _download_series(ticker: str, start, end, name: str) -> pd.Series:
    try:
        ser = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)['Close']
        ser.name = name
        return ser
    except Exception:
        return pd.Series(dtype=float, name=name)


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _compute_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """Calculate MACD, Signal, and Histogram"""
    exp1 = prices.ewm(span=fast_period, adjust=False).mean()
    exp2 = prices.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def _build_features(df: pd.DataFrame, start_date, end_date) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()

    # Flatten yfinance multi-index if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Use Adj Close for returns; keep Close for level features
    if 'Adj Close' in data.columns:
        data['AdjClose'] = data['Adj Close']
    else:
        data['AdjClose'] = data['Close']

    # Context features: CAC40 & VIX
    cac = _download_series('^FCHI', start_date, end_date, 'CAC')
    vix = _download_series('^VIX',  start_date, end_date, 'VIX')
    eurostoxx = _download_series('^STOXX', start_date, end_date, 'STOXX')
    data = data.join([cac, vix, eurostoxx], how='left')

    # Returns & lags
    data['Ret1'] = _pct_change_safe(data['AdjClose'], 1)
    data['Ret5'] = _pct_change_safe(data['AdjClose'], 5)
    data['Ret20'] = _pct_change_safe(data['AdjClose'], 20)
    for k in (1, 2, 3, 5, 10):
        data[f'LagRet{k}'] = data['Ret1'].shift(k)

    # Moving averages & momentum
    data['MA10'] = data['AdjClose'].rolling(10).mean()
    data['MA30'] = data['AdjClose'].rolling(30).mean()
    data['MA60'] = data['AdjClose'].rolling(60).mean()
    data['MA200'] = data['AdjClose'].rolling(200).mean()
    data['Momentum10'] = data['AdjClose'] - data['AdjClose'].shift(10)
    data['Volatility10'] = data['Ret1'].rolling(10).std()
    data['Volatility20'] = data['Ret1'].rolling(20).std()
    
    # MA crossovers - binary indicators
    data['MA_10_30_Cross'] = (data['MA10'] > data['MA30']).astype(int)
    data['MA_10_60_Cross'] = (data['MA10'] > data['MA60']).astype(int)

    # RSI - overbought/oversold indicator
    data['RSI14'] = _compute_rsi(data['AdjClose'])
    data['RSI_Overbought'] = (data['RSI14'] > 70).astype(int)
    data['RSI_Oversold'] = (data['RSI14'] < 30).astype(int)

    # MACD - trend following momentum indicator
    macd, signal, hist = _compute_macd(data['AdjClose'])
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    data['MACD_Hist'] = hist
    data['MACD_Cross'] = ((data['MACD'] > data['MACD_Signal']).astype(int) - 
                          (data['MACD_Signal'] > data['MACD']).astype(int))

    # Bollinger bands on MA20 / Volatility20
    vol20 = data['Ret1'].rolling(20).std()
    ma20 = data['AdjClose'].rolling(20).mean()
    data['BBU'] = ma20 + 2 * vol20
    data['BBL'] = ma20 - 2 * vol20
    data['BB_Width'] = (data['BBU'] - data['BBL']) / ma20  # Normalized BB width
    data['BB_Position'] = (data['AdjClose'] - data['BBL']) / (data['BBU'] - data['BBL'])

    # OBV
    data['OBV'] = _compute_obv_from_series(data['AdjClose'], data['Volume'])
    data['OBV_MA10'] = data['OBV'].rolling(10).mean()
    data['OBV_Trend'] = (data['OBV'] > data['OBV_MA10']).astype(int)

    # Stochastic oscillator
    low14 = data['Low'].rolling(window=14).min()
    high14 = data['High'].rolling(window=14).max()
    denom = (high14 - low14).replace(0, np.nan)
    data['StochK'] = 100 * ((data['AdjClose'] - low14) / denom)
    data['StochD'] = data['StochK'].rolling(window=3).mean()
    data['Stoch_Cross'] = ((data['StochK'] > data['StochD']).astype(int) - 
                          (data['StochD'] > data['StochK']).astype(int))

    # Volume + context dynamics
    data['VolChg'] = _pct_change_safe(data['Volume']).fillna(0)
    data['Vol_SMA5'] = data['Volume'].rolling(5).mean()
    data['Vol_Ratio'] = data['Volume'] / data['Vol_SMA5']
    
    # Market context indicators
    if 'CAC' in data:
        data['CAC_ret1'] = _pct_change_safe(data['CAC'])
        data['CAC_ret5'] = _pct_change_safe(data['CAC'], 5)
        data['CAC_vol10'] = _pct_change_safe(data['CAC']).rolling(10).std()
    if 'VIX' in data:
        data['VIX_chg'] = _pct_change_safe(data['VIX'])
        data['VIX_MA10'] = data['VIX'].rolling(10).mean()
        # Market sentiment based on VIX
        data['VIX_Regime'] = (data['VIX'] > data['VIX_MA10']).astype(int)
    if 'STOXX' in data:
        data['STOXX_ret1'] = _pct_change_safe(data['STOXX'])
        data['STOXX_ret5'] = _pct_change_safe(data['STOXX'], 5)

    # Volatility regime features
    data['Vol_Regime'] = (data['Volatility20'] > data['Volatility20'].rolling(50).mean()).astype(int)
    
    # Calendar features
    data['DayOfWeek'] = pd.to_datetime(data.index).dayofweek
    data['Month'] = pd.to_datetime(data.index).month

    data = data.dropna().copy()

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'AdjClose', 'Ret1', 'Ret5', 'Ret20', 'LagRet1', 'LagRet2', 'LagRet3', 'LagRet5', 'LagRet10',
        'MA10', 'MA30', 'MA60', 'MA200', 'Momentum10', 'Volatility10', 'Volatility20',
        'MA_10_30_Cross', 'MA_10_60_Cross', 'RSI14', 'RSI_Overbought', 'RSI_Oversold',
        'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Cross',
        'BBU', 'BBL', 'BB_Width', 'BB_Position',
        'OBV', 'OBV_MA10', 'OBV_Trend',
        'StochK', 'StochD', 'Stoch_Cross',
        'VolChg', 'Vol_SMA5', 'Vol_Ratio', 'Vol_Regime',
        'DayOfWeek', 'Month'
    ]
    
    # Add market context features if available
    if 'CAC_ret1' in data:
        features.extend(['CAC_ret1', 'CAC_ret5', 'CAC_vol10'])
    if 'VIX_chg' in data:
        features.extend(['VIX_chg', 'VIX_MA10', 'VIX_Regime'])
    if 'STOXX_ret1' in data:
        features.extend(['STOXX_ret1', 'STOXX_ret5'])

    return data, features

# -----------------------------
# Model building
# -----------------------------

def _build_ensemble_classifier(random_state=42):
    """Create an ensemble classifier combining multiple algorithms"""
    # Base models
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state)
    hgb = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=200, random_state=random_state)
    svc = SVC(probability=True, random_state=random_state)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('hgb', hgb),
            ('svc', svc)
        ],
        voting='soft'
    )
    
    return ensemble

def _build_stacked_classifier(random_state=42):
    """Create a stacked classifier for better performance"""
    # Base models
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ('hgb', HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=random_state)),
        ('svc', SVC(probability=True, random_state=random_state))
    ]
    
    # Final estimator
    final_estimator = LogisticRegression(random_state=random_state)
    
    # Stacking classifier
    stacked = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=3,
        stack_method='predict_proba'
    )
    
    return stacked

# -----------------------------
# Backtest utilities
# -----------------------------

def _signals_from_proba(p: np.ndarray, threshold: float, deadband: float) -> np.ndarray:
    """Map probabilities to *desired* signals: -1, 0, +1.
    0 inside the deadband around 0.5; +1 if >= threshold; -1 otherwise.
    """
    p = np.asarray(p)
    sig = np.zeros_like(p)
    centered = p - 0.5
    neutral = np.abs(centered) < deadband
    # For non-neutral, choose side by threshold; below threshold => DOWN
    sig[~neutral] = np.where(p[~neutral] >= threshold, 1, -1)
    return sig  # -1,0,+1 (desired instantaneous signal)


def _adaptive_threshold(vol: float, base_threshold: float, vol_scaling: bool = True) -> float:
    """Adjust threshold based on volatility - higher threshold in high volatility periods"""
    if not vol_scaling or np.isnan(vol):
        return base_threshold
    
    # Scale threshold between base and 0.65 based on volatility (assumes vol is normalized)
    adjusted = base_threshold + min(0.08, max(0, vol * 0.15))
    return min(0.75, adjusted)  # Cap at 0.75

def _backtest_walk_forward(X: pd.DataFrame, y_ret: pd.Series, context: pd.DataFrame, clf_builder, threshold: float, deadband: float,
                           cost_bps: float, calibrate: bool, long_only: bool, regime_ma: int, vol_target: float,
                           horizon: int, random_state: int = 42):
    """Walk‑forward backtest that *persists* positions and charges costs on position changes."""
    n_splits = max(3, min(8, len(X) // 80))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)

    all_idx = []
    all_proba = []

    for tr, te in tscv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr = y_ret.iloc[tr]

        # Feature selection to focus on most predictive features
        if len(X_tr) > 100:  # Ensure enough data for feature selection
            # FIX: Use RandomForest for feature selection as it's more reliable
            rf_selector = RandomForestClassifier(n_estimators=50, random_state=random_state)
            rf_selector.fit(X_tr, (y_tr > 0).astype(int))
            
            # Get feature importances and select top features
            importances = rf_selector.feature_importances_
            n_features = min(len(X_tr.columns), max(10, len(X_tr.columns) // 2))
            top_indices = np.argsort(importances)[-n_features:]
            selected_features = X_tr.columns[top_indices]
            
            X_tr = X_tr[selected_features]
            X_te = X_te[selected_features]

        clf = clf_builder()
        if calibrate:
            clf = CalibratedClassifierCV(clf, cv=3, method='isotonic')
        clf.fit(X_tr, (y_tr > 0).astype(int))
        proba = clf.predict_proba(X_te)[:, 1]
        all_idx.append(X_te.index)
        all_proba.append(proba)

    if not all_proba:
        return {
            'cum_return': float('nan'),
            'sharpe_annual': float('nan'),
            'hit_ratio': float('nan'),
            'max_drawdown': float('nan'),
            'trades': 0,
            'buyhold_cum_return': float('nan')
        }

    # Combine out-of-fold predictions
    proba_oof = np.concatenate(all_proba)
    idx_oof = np.concatenate([np.array(ix) for ix in all_idx])
    order = np.argsort(idx_oof)
    idx_sorted = idx_oof[order]
    proba_sorted = proba_oof[order]

    # Use non-overlapping steps of length = horizon to avoid overlapping forward returns
    step = max(1, int(horizon))
    keep_idx = np.arange(0, len(proba_sorted), step)
    idx_sorted = idx_sorted[keep_idx]
    proba_sorted = proba_sorted[keep_idx]

    # Get context for these indices
    ctx = context.loc[pd.Index(idx_sorted)]
    px = ctx['AdjClose']
    
    # Get volatility for adaptive thresholding
    realized_vol = ctx['Ret1'].fillna(0.0).rolling(20).std().replace(0, np.nan).bfill().ffill()
    vol_z = (realized_vol - realized_vol.rolling(100, min_periods=50).mean()) / realized_vol.rolling(100, min_periods=50).std()
    vol_z = vol_z.fillna(0)

    # Apply adaptive thresholds based on volatility
    adaptive_thresholds = [_adaptive_threshold(v, threshold, True) for v in vol_z]
    
    # Desired signals from probability with adaptive thresholds
    desired = np.zeros_like(proba_sorted)
    for i in range(len(proba_sorted)):
        p = proba_sorted[i]
        t = adaptive_thresholds[i]
        if abs(p - 0.5) < deadband:
            desired[i] = 0  # Neutral zone
        else:
            desired[i] = 1 if p >= t else -1  # Apply adaptive threshold

    # Regime filter using AdjClose vs MA(regime_ma)
    regime = (px > px.rolling(regime_ma).mean()).astype(int)  # 1 if up regime

    # If long_only: suppress shorts; else, allow both sides
    if long_only:
        desired = np.where(desired > 0, 1, 0)
    else:
        # Suppress longs in down regime and shorts in up regime
        desired = np.where((regime == 1) & (desired < 0), 0, desired)
        desired = np.where((regime == 0) & (desired > 0), 0, desired)

    # Discrete SIDE in {-1,0,+1} that we will scale later
    side = np.zeros_like(desired, dtype=float)
    for i in range(len(side)):
        side[i] = 0.0 if desired[i] == 0 else float(desired[i])

    # Volatility targeting: scale position magnitude by target / realized_vol
    ret1 = ctx['Ret1'].fillna(0.0)
    realized_vol = ret1.rolling(20).std().replace(0, np.nan).bfill().ffill()
    scale = (vol_target / realized_vol).clip(upper=1.0)
    pos = side * scale.values

    # Convert forward log-returns to simple returns for block P&L
    y_block = y_ret.loc[pd.Index(idx_sorted)].values  # forward log-return over `horizon`
    y_simple_block = np.expm1(y_block)

    # Apply position *on the same block* (we trade block-to-block; entries at block boundaries)
    strat_ret_gross = pos * y_simple_block

    # Transaction costs when SIDE changes between blocks (per side). Flip -1->+1 counts as 2.
    side_s = pd.Series(side, index=pd.Index(idx_sorted), dtype=float)
    delta_side = side_s.diff().fillna(side_s.iloc[0]).abs().values
    cost_rate = cost_bps / 10000.0
    strat_ret_net = strat_ret_gross - cost_rate * delta_side

    # Buy&Hold baseline on the same non-overlapping blocks (simple returns)
    bh_simple = y_simple_block
    bh_cum = np.cumprod(1 + bh_simple)
    bh_cum_return = float(bh_cum[-1] - 1) if len(bh_cum) else float('nan')

    # Metrics on net returns
    if len(strat_ret_net) == 0:
        return {
            'cum_return': float('nan'),
            'sharpe_annual': float('nan'),
            'hit_ratio': float('nan'),
            'max_drawdown': float('nan'),
            'trades': 0,
            'buyhold_cum_return': bh_cum_return
        }

    cum = np.cumprod(1 + strat_ret_net)
    cum_return = float(cum[-1] - 1)
    mu = np.mean(strat_ret_net)
    sd = np.std(strat_ret_net, ddof=1) if len(strat_ret_net) > 1 else np.nan
    sharpe_annual = float(np.sqrt(252) * mu / sd) if (sd and sd > 0) else float('nan')
    hit_ratio = float(np.mean(strat_ret_net > 0))

    # Max drawdown
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    max_dd = float(np.min(dd)) if len(dd) else float('nan')

    # Count executed trades as number of nonzero SIDE changes (delta_side>0); flip counts as 2
    trades = int(delta_side.sum())

    return {
        'cum_return': cum_return,
        'sharpe_annual': sharpe_annual,
        'hit_ratio': hit_ratio,
        'max_drawdown': max_dd,
        'trades': trades,
        'buyhold_cum_return': bh_cum_return
    }


# -----------------------------
# Main pipeline
# -----------------------------

# Import the results formatter
try:
    from .results_formatter import ResultsFormatter
except ImportError:
    try:
        # For direct execution
        from results_formatter import ResultsFormatter
    except ImportError:
        # Define a simple fallback formatter
        class ResultsFormatter:
            @staticmethod
            def format_cli(prediction):
                return json.dumps(prediction, indent=2)
            
            @staticmethod
            def format_json(prediction):
                return prediction


def run(ticker: str = "ALO.PA", years: float = 5.0, horizon: int = 5, threshold: float = 0.60, deadband: float = 0.10,
        cost_bps: float = 10.0, clf_model: str = 'histgb', calibrate: bool = True, long_only: bool = True,
        regime_ma: int = 100, vol_target: float = 0.01, outdir: str | None = None, plot: bool = False, random_state: int = 42,
        use_ensemble: bool = True):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=int(years * 365))

    # Use auto_adjust to account for splits/dividends; keep unadjusted Close as level feature
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data is None or len(data) < 200:
        raise RuntimeError(f"Not enough data for {ticker}. Got {0 if data is None else len(data)} rows.")

    data, features = _build_features(data, start_date, end_date)

    # Target: forward **log-return** over horizon (stabilizes training & avoids compounding bias)
    y_ret = _log_forward_return(data['AdjClose'], horizon).dropna()
    # Align features with target
    X = data.loc[y_ret.index, features]
    # Context for backtest/regime/vol targeting
    context = data.loc[y_ret.index, ['AdjClose', 'Ret1']].copy()

    # Feature selection to keep only most informative features
    if len(X) > 100:  # Ensure we have enough data
        base_model = RandomForestRegressor(n_estimators=200, random_state=random_state)
        base_model.fit(X, y_ret)
        importances = pd.Series(base_model.feature_importances_, index=X.columns)
        top_features = importances.nlargest(min(30, len(importances))).index.tolist()
        X_reduced = X[top_features]
    else:
        X_reduced = X

    # Models
    reg_model = RandomForestRegressor(random_state=random_state, n_estimators=500, max_depth=None, n_jobs=-1)
    
    # Choose classifier model
    if use_ensemble:
        base_clf = lambda: _build_ensemble_classifier(random_state=random_state)
    elif clf_model == 'stacked':
        base_clf = lambda: _build_stacked_classifier(random_state=random_state)
    elif clf_model == 'histgb':
        base_clf = lambda: HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=300, random_state=random_state)
    else:
        base_clf = lambda: RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)

    # Time-aware CV for metrics
    n_splits = max(3, min(8, len(X) // 80))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)

    maes, rmses, accs, precs, recs, f1s, aucs = [], [], [], [], [], [], []

    for tr, te in tscv.split(X_reduced):
        X_tr, X_te = X_reduced.iloc[tr], X_reduced.iloc[te]
        yr_tr, yr_te = y_ret.iloc[tr], y_ret.iloc[te]

        # Regression on returns
        reg_model.fit(X_tr, yr_tr)
        pred_ret = reg_model.predict(X_te)
        maes.append(mean_absolute_error(yr_te, pred_ret))
        rmses.append(np.sqrt(mean_squared_error(yr_te, pred_ret)))

        # Classification on sign(returns)
        clf = base_clf()
        if calibrate:
            clf = CalibratedClassifierCV(clf, cv=3, method='isotonic')
        clf.fit(X_tr, (yr_tr > 0).astype(int))
        pred_lbl = clf.predict(X_te)
        
        # Fix: Use yr_te instead of yr_tr for test metrics
        y_te_bin = (yr_te > 0).astype(int)
        accs.append(accuracy_score(y_te_bin, pred_lbl))
        precs.append(precision_score(y_te_bin, pred_lbl, zero_division=0))
        recs.append(recall_score(y_te_bin, pred_lbl, zero_division=0))
        f1s.append(f1_score(y_te_bin, pred_lbl, zero_division=0))
        
        try:
            proba = clf.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te_bin, proba))
        except Exception:
            aucs.append(float('nan'))

    # Retrain on full history
    reg_model.fit(X_reduced, y_ret)
    clf = base_clf()
    if calibrate:
        clf = CalibratedClassifierCV(clf, cv=3, method='isotonic')
    clf.fit(X_reduced, (y_ret > 0).astype(int))

    # Latest prediction
    latest_features = X_reduced.iloc[[-1]]
    last_close = float(data.loc[X.index[-1], 'AdjClose'])

    pred_ret_latest = float(reg_model.predict(latest_features)[0])
    proba_up = float(getattr(clf, 'predict_proba')(latest_features)[0, 1]) if hasattr(clf, 'predict_proba') else float('nan')

    # Get current volatility for adaptive thresholding
    latest_vol = data['Ret1'].tail(20).std()
    vol_z = (latest_vol - data['Ret1'].tail(100).std()) / data['Ret1'].tail(100).std()
    adaptive_threshold = _adaptive_threshold(vol_z, threshold)
    
    # Decision rule with adaptive threshold and deadband
    if abs(proba_up - 0.5) < deadband:
        decision = 'NO_TRADE'
    else:
        decision = 'UP' if proba_up >= adaptive_threshold else 'DOWN'

    # pred_ret_latest is a log-return → price = last_close * exp(logret)
    predicted_price = float(last_close * np.exp(pred_ret_latest))

    # Walk-forward backtest with rule
    backtest = _backtest_walk_forward(
        X_reduced, y_ret, context, base_clf, threshold, deadband, cost_bps, calibrate, long_only, regime_ma, vol_target,
        horizon=horizon, random_state=random_state
    )
    bh_cum_ret = backtest.get('buyhold_cum_return')
    metrics = {
        'reg_mae_cv_mean': float(np.nanmean(maes)),
        'reg_rmse_cv_mean': float(np.nanmean(rmses)),
        'clf_accuracy_cv_mean': float(np.nanmean(accs)),
        'clf_precision_cv_mean': float(np.nanmean(precs)),
        'clf_recall_cv_mean': float(np.nanmean(recs)),
        'clf_f1_cv_mean': float(np.nanmean(f1s)),
        'clf_auc_cv_mean': float(np.nanmean(aucs)),
        'cv_splits': int(n_splits),
        'n_obs': int(len(X)),
        'backtest': backtest,
        'buyhold_cum_return': bh_cum_ret
    }

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
        'cost_bps': cost_bps,
        'clf_model': 'ensemble' if use_ensemble else clf_model,
        'calibrated': calibrate,
        'metrics': metrics,
        'long_only': long_only,
        'regime_ma': regime_ma,
        'vol_target': vol_target,
        'top_features': top_features if len(X) > 100 else features[:10],
    }

    # Save results if outdir specified
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        
        # Save raw prediction
        with open(os.path.join(outdir, "latest_prediction.json"), "w") as f:
            json.dump(payload, f, indent=2)
        
        # Save enhanced results
        enhanced_payload = ResultsFormatter.format_json(payload)
        with open(os.path.join(outdir, "latest_results.json"), "w") as f:
            json.dump(enhanced_payload, f, indent=2)
        
        # Save simple CSV snapshot
        snap = data.tail(1).copy()
        snap["Predicted_Pct_Return"] = pred_ret_latest
        snap["Predicted_Price"] = predicted_price
        snap["Decision"] = decision
        snap.to_csv(os.path.join(outdir, "latest_prediction.csv"))

    # Generate plots
    if plot:
        try:
            import matplotlib.pyplot as plt
            # Create a more comprehensive visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
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
            ax1.set_title(f"{ticker} Price Prediction - Next {horizon} Days", fontsize=16)
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
                plt.savefig(os.path.join(outdir, f"{ticker}_prediction.png"), dpi=100, bbox_inches='tight')
            
            plt.show()
        except Exception as e:
            warnings.warn(f"Plotting failed: {e}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alstom stock predictor with returns target, market context, and JSON output")
    parser.add_argument("--ticker", default="ALO.PA", help="Ticker (default: ALO.PA)")
    parser.add_argument("--years", type=float, default=5.0, help="Lookback window in years (default: 5)")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.57, help="UP decision threshold on proba_up (default: 0.57)")
    parser.add_argument("--deadband", type=float, default=0.05, help="Neutral zone around 0.5 (default: 0.05)")
    parser.add_argument("--cost_bps", type=float, default=10.0, help="Per-trade cost in bps (default: 10)")
    parser.add_argument("--clf_model", choices=["histgb", "rf", "stacked"], default="histgb", help="Classifier model")
    parser.add_argument("--calibrate", action="store_true", default=True, help="Calibrate classifier probabilities (isotonic)")
    parser.add_argument("--long_only", action="store_true", default=True, help="Only take long signals (suppress shorts)")
    parser.add_argument("--regime_ma", type=int, default=100, help="MA window (days) for regime filter")
    parser.add_argument("--vol_target", type=float, default=0.01, help="Daily volatility target for position sizing (e.g., 0.01 = 1%)")
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory for artifacts")
    parser.add_argument("--plot", action="store_true", help="Show plot (off by default)")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble model (off by default)")

    args = parser.parse_args()
    payload = run(
        ticker=args.ticker,
        years=args.years,
        horizon=args.horizon,
        threshold=args.threshold,
        deadband=args.deadband,
        cost_bps=args.cost_bps,
        clf_model=args.clf_model,
        calibrate=args.calibrate,
        long_only=args.long_only,
        regime_ma=args.regime_ma,
        vol_target=args.vol_target,
        outdir=args.outdir,
        plot=args.plot,
        use_ensemble=args.ensemble,
    )
    
    # Print results in a more user-friendly format
    formatted_output = ResultsFormatter.format_cli(payload)
    print(formatted_output)
