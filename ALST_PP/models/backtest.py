"""
Backtesting functionality for stock prediction models
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from .models import create_model

def run_backtest(X, y_ret, context, model_type, threshold, deadband, 
               cost_bps, calibrate, long_only, regime_ma, vol_target, 
               horizon, random_state=42):
    """Run a walk-forward backtest of the trading strategy"""
    # Create model factory function
    clf_builder = lambda: create_model(model_type, random_state)
    
    # Run the backtest
    return _backtest_walk_forward(
        X, y_ret, context, clf_builder, threshold, deadband, 
        cost_bps, calibrate, long_only, regime_ma, vol_target, 
        horizon, random_state
    )

def _signals_from_proba(p: np.ndarray, threshold: float, deadband: float) -> np.ndarray:
    """Map probabilities to *desired* signals: -1, 0, +1."""
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
    
    # Scale threshold between base and 0.65 based on volatility
    adjusted = base_threshold + min(0.08, max(0, vol * 0.15))
    return min(0.75, adjusted)  # Cap at 0.75

def _backtest_walk_forward(X: pd.DataFrame, y_ret: pd.Series, context: pd.DataFrame, clf_builder, threshold: float, deadband: float,
                           cost_bps: float, calibrate: bool, long_only: bool, regime_ma: int, vol_target: float,
                           horizon: int, random_state: int = 42):
    """
    Walk‑forward backtest that *persists* positions and charges costs on position changes.
    - Build out-of-fold probabilities using TimeSeriesSplit
    - Convert to desired signals via threshold/deadband
    - Create *actual* positions that persist (carry last non-neutral)
    - Apply next-period returns to today's position (no look‑ahead)
    - Apply costs only when position changes; flipping -1->+1 counts as 2 changes
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    n_splits = max(3, min(8, len(X) // 80))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)

    all_idx = []
    all_proba = []

    for tr, te in tscv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr = y_ret.iloc[tr]

        # Feature selection to focus on most predictive features
        if len(X_tr) > 100:  # Ensure enough data for feature selection
            base_clf = HistGradientBoostingClassifier(random_state=random_state)
            selector = SelectFromModel(estimator=base_clf, threshold='mean')
            selector.fit(X_tr, (y_tr > 0).astype(int))
            selected_features = X_tr.columns[selector.get_support()]
            if len(selected_features) >= 10:  # Ensure we have enough features
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
            'trades': 0
        }

    # Combine out-of-fold predictions
    proba_oof = np.concatenate(all_proba)
    idx_oof = np.concatenate([np.array(ix) for ix in all_idx])
    
    # Sort by index
    order = np.argsort(idx_oof)
    idx_sorted = idx_oof[order]
    proba_sorted = proba_oof[order]

    # Use non-overlapping steps of length = horizon
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

    # Apply adaptive thresholds
    adaptive_thresholds = [_adaptive_threshold(v, threshold, True) for v in vol_z]
    
    # Generate signals
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

    # Apply strategy rules
    if long_only:
        desired = np.where(desired > 0, 1, 0)
    else:
        # Suppress longs in down regime and shorts in up regime
        desired = np.where((regime == 1) & (desired < 0), 0, desired)
        desired = np.where((regime == 0) & (desired > 0), 0, desired)

    # Convert to position sizes
    side = np.zeros_like(desired, dtype=float)
    for i in range(len(side)):
        side[i] = 0.0 if desired[i] == 0 else float(desired[i])

    # Apply volatility targeting for position sizing
    ret1 = ctx['Ret1'].fillna(0.0)
    realized_vol = ret1.rolling(20).std().replace(0, np.nan).bfill().ffill()
    scale = (vol_target / realized_vol).clip(upper=1.0)
    pos = side * scale.values

    # Calculate returns
    y_block = y_ret.loc[pd.Index(idx_sorted)].values
    y_simple_block = np.expm1(y_block)
    strat_ret_gross = pos * y_simple_block

    # Apply transaction costs
    side_s = pd.Series(side, index=pd.Index(idx_sorted), dtype=float)
    delta_side = side_s.diff().fillna(side_s.iloc[0]).abs().values
    cost_rate = cost_bps / 10000.0
    strat_ret_net = strat_ret_gross - cost_rate * delta_side

    # Calculate buy & hold returns
    bh_simple = y_simple_block
    bh_cum = np.cumprod(1 + bh_simple)
    bh_cum_return = float(bh_cum[-1] - 1) if len(bh_cum) else float('nan')

    # Calculate strategy metrics
    if len(strat_ret_net) == 0:
        return {
            'cum_return': float('nan'),
            'sharpe_annual': float('nan'),
            'hit_ratio': float('nan'),
            'max_drawdown': float('nan'),
            'trades': 0
        }

    cum = np.cumprod(1 + strat_ret_net)
    cum_return = float(cum[-1] - 1)
    mu = np.mean(strat_ret_net)
    sd = np.std(strat_ret_net, ddof=1) if len(strat_ret_net) > 1 else np.nan
    sharpe_annual = float(np.sqrt(252) * mu / sd) if (sd and sd > 0) else float('nan')
    hit_ratio = float(np.mean(strat_ret_net > 0))

    # Calculate max drawdown
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    max_dd = float(np.min(dd)) if len(dd) else float('nan')

    # Count trades
    trades = int(delta_side.sum())

    return {
        'cum_return': cum_return,
        'sharpe_annual': sharpe_annual,
        'hit_ratio': hit_ratio,
        'max_drawdown': max_dd,
        'trades': trades,
        'buyhold_cum_return': bh_cum_return
    }
