"""
Feature engineering for stock price prediction
"""
import numpy as np
import pandas as pd
import yfinance as yf

def download_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    return yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

def build_features(df: pd.DataFrame, start_date, end_date) -> tuple[pd.DataFrame, list[str]]:
    """Build features for prediction model"""
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

def _compute_obv_from_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On Balance Volume"""
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
    """Calculate percent change with safety for division errors"""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = s.pct_change(periods=periods)
    return r.replace([np.inf, -np.inf], np.nan)

def log_forward_return(s: pd.Series, horizon: int) -> pd.Series:
    """Compute forward log-return over `horizon` days: ln(P_{t+h}/P_t) aligned at t."""
    ln = np.log(s.astype(float))
    return (ln.shift(-horizon) - ln)

def _download_series(ticker: str, start, end, name: str) -> pd.Series:
    """Download a single series from Yahoo Finance"""
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
