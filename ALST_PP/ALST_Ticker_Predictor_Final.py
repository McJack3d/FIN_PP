import argparse
import json
import os
import warnings
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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


def _download_series(ticker: str, start, end, name: str) -> pd.Series:
    try:
        ser = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)['Close']
        ser.name = name
        return ser
    except Exception:
        return pd.Series(dtype=float, name=name)


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
    data = data.join([cac, vix], how='left')

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
    data['Momentum10'] = data['AdjClose'] - data['AdjClose'].shift(10)
    data['Volatility10'] = data['Ret1'].rolling(10).std()

    # Bollinger bands on MA20 / Volatility20
    vol20 = data['Ret1'].rolling(20).std()
    ma20 = data['AdjClose'].rolling(20).mean()
    data['BBU'] = ma20 + 2 * vol20
    data['BBL'] = ma20 - 2 * vol20

    # OBV
    data['OBV'] = _compute_obv_from_series(data['AdjClose'], data['Volume'])

    # Stochastic oscillator
    low14 = data['Low'].rolling(window=14).min()
    high14 = data['High'].rolling(window=14).max()
    denom = (high14 - low14).replace(0, np.nan)
    data['StochK'] = 100 * ((data['AdjClose'] - low14) / denom)
    data['StochD'] = data['StochK'].rolling(window=3).mean()

    # Volume + context dynamics
    data['VolChg'] = _pct_change_safe(data['Volume']).fillna(0)
    if 'CAC' in data:
        data['CAC_ret1'] = _pct_change_safe(data['CAC'])
    if 'VIX' in data:
        data['VIX_chg'] = _pct_change_safe(data['VIX'])

    data = data.dropna().copy()

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'AdjClose', 'Ret1', 'Ret5', 'Ret20', 'LagRet1', 'LagRet2', 'LagRet3', 'LagRet5', 'LagRet10',
        'MA10', 'MA30', 'MA60', 'Momentum10', 'Volatility10', 'BBU', 'BBL',
        'OBV', 'StochK', 'StochD', 'VolChg'
    ]
    if 'CAC_ret1' in data:
        features.append('CAC_ret1')
    if 'VIX_chg' in data:
        features.append('VIX_chg')

    return data, features


# -----------------------------
# Backtest utilities
# -----------------------------

def _signals_from_proba(p: np.ndarray, threshold: float, deadband: float) -> np.ndarray:
    # p ∈ [0,1]. deadband around 0.5; threshold for UP decision
    sig = np.zeros_like(p)
    centered = p - 0.5
    neutral = np.abs(centered) < deadband
    sig[~neutral] = np.where(p[~neutral] >= threshold, 1, -1)
    return sig  # -1, 0, +1


def _max_drawdown(cumret: np.ndarray) -> float:
    peak = -np.inf
    max_dd = 0.0
    for x in cumret:
        if x > peak:
            peak = x
        dd = (x/peak - 1.0) if peak != -np.inf else 0.0
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def _backtest_walk_forward(X: pd.DataFrame, y_ret: pd.Series, clf_builder, threshold: float, deadband: float, cost_bps: float, calibrate: bool, random_state: int = 42):
    # Walk-forward: build out-of-fold probabilities on test slices
    n_splits = max(3, min(8, len(X) // 80))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_idx = []
    all_proba = []

    for tr, te in tscv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y_ret.iloc[tr], y_ret.iloc[te]

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

    proba_oof = np.concatenate(all_proba)
    idx_oof = np.concatenate([np.array(ix) for ix in all_idx])
    order = np.argsort(idx_oof)
    proba_oof = proba_oof[order]
    y_oof = y_ret.loc[idx_oof[order]].values

    # Signals and returns
    sig = _signals_from_proba(proba_oof, threshold, deadband)  # -1,0,+1
    # Trading rule: apply position for the next period return
    strat_ret = sig * y_oof

    # Costs: apply cost once per non-neutral decision
    cost = (cost_bps / 10000.0)
    trades = np.count_nonzero(sig)
    strat_ret_after_cost = strat_ret - cost * (sig != 0)

    # Metrics
    cum = np.cumprod(1 + strat_ret_after_cost)
    cum_return = float(cum[-1] - 1) if len(cum) else float('nan')
    mu = np.mean(strat_ret_after_cost)
    sd = np.std(strat_ret_after_cost, ddof=1) if len(strat_ret_after_cost) > 1 else np.nan
    sharpe_annual = float(np.sqrt(252) * mu / sd) if (sd and sd > 0) else float('nan')
    hit_ratio = float(np.mean(strat_ret_after_cost > 0)) if len(strat_ret_after_cost) else float('nan')
    max_dd = _max_drawdown(cum) if len(cum) else float('nan')

    return {
        'cum_return': cum_return,
        'sharpe_annual': sharpe_annual,
        'hit_ratio': hit_ratio,
        'max_drawdown': max_dd,
        'trades': int(trades)
    }


# -----------------------------
# Main pipeline
# -----------------------------

def run(ticker: str = "ALO.PA", years: float = 5.0, horizon: int = 5, threshold: float = 0.57, deadband: float = 0.05,
        cost_bps: float = 10.0, clf_model: str = 'histgb', calibrate: bool = False, outdir: str | None = None, plot: bool = False, random_state: int = 42):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=int(years * 365))

    # Use auto_adjust to account for splits/dividends; keep unadjusted Close as level feature
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data is None or len(data) < 200:
        raise RuntimeError(f"Not enough data for {ticker}. Got {0 if data is None else len(data)} rows.")

    data, features = _build_features(data, start_date, end_date)

    # Target: forward return over horizon
    y_ret = _pct_change_safe(data['AdjClose'], horizon).shift(-horizon).dropna()
    # Align features with target
    X = data.loc[y_ret.index, features]

    # Models
    reg_model = RandomForestRegressor(random_state=random_state, n_estimators=400, max_depth=None, n_jobs=-1)
    if clf_model == 'histgb':
        base_clf = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=random_state)
    else:
        base_clf = lambda: RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)

    # Time-aware CV for metrics
    n_splits = max(3, min(8, len(X) // 80))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    maes, rmses, accs, precs, recs, f1s, aucs = [], [], [], [], [], [], []

    for tr, te in tscv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
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
        accs.append(accuracy_score((yr_te > 0).astype(int), pred_lbl))
        precs.append(precision_score((yr_te > 0).astype(int), pred_lbl, zero_division=0))
        recs.append(recall_score((yr_te > 0).astype(int), pred_lbl, zero_division=0))
        f1s.append(f1_score((yr_te > 0).astype(int), pred_lbl, zero_division=0))
        try:
            proba = clf.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score((yr_te > 0).astype(int), proba))
        except Exception:
            aucs.append(float('nan'))

    # Retrain on full history
    reg_model.fit(X, y_ret)
    clf = base_clf()
    if calibrate:
        clf = CalibratedClassifierCV(clf, cv=3, method='isotonic')
    clf.fit(X, (y_ret > 0).astype(int))

    # Latest prediction
    latest_features = X.iloc[[-1]]
    last_close = float(data.loc[X.index[-1], 'AdjClose'])

    pred_ret_latest = float(reg_model.predict(latest_features)[0])
    proba_up = float(getattr(clf, 'predict_proba')(latest_features)[0, 1]) if hasattr(clf, 'predict_proba') else float('nan')

    # Decision rule with threshold/deadband
    if abs(proba_up - 0.5) < deadband:
        decision = 'NO_TRADE'
    else:
        decision = 'UP' if proba_up >= threshold else 'DOWN'

    predicted_price = float(last_close * (1.0 + pred_ret_latest))

    # Walk-forward backtest with rule
    backtest = _backtest_walk_forward(X, y_ret, base_clf, threshold, deadband, cost_bps, calibrate, random_state=random_state)

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
        'backtest': backtest
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
        'deadband': deadband,
        'cost_bps': cost_bps,
        'clf_model': clf_model,
        'calibrated': calibrate,
        'metrics': metrics,
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "latest_prediction.json"), "w") as f:
            json.dump(payload, f, indent=2)
        snap = data.tail(1).copy()
        snap["Predicted_Pct_Return"] = pred_ret_latest
        snap["Predicted_Price"] = predicted_price
        snap["Decision"] = decision
        snap.to_csv(os.path.join(outdir, "latest_prediction.csv"))

    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            tail = data.tail(180)
            plt.plot(tail['AdjClose'], label='Adj Close')
            plt.plot(tail['MA10'], label='MA10', linestyle='--')
            plt.plot(tail['BBU'], label='BB Upper', linestyle='--', alpha=0.5)
            plt.plot(tail['BBL'], label='BB Lower', linestyle='--', alpha=0.5)
            plt.axhline(predicted_price, linestyle='--', label=f'Predicted: €{predicted_price:.2f}')
            plt.title(f"{ticker} (last 180d) + next-{horizon}d prediction")
            plt.xlabel("Date"); plt.ylabel("Price")
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
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
    parser.add_argument("--clf_model", choices=["histgb", "rf"], default="histgb", help="Classifier model")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate classifier probabilities (isotonic)")
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory for artifacts")
    parser.add_argument("--plot", action="store_true", help="Show plot (off by default)")

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
        outdir=args.outdir,
        plot=args.plot,
    )
    print(json.dumps(payload))
