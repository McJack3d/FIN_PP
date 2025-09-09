
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import matplotlib.pyplot as plt
import os

# 1. Load Alstom stock data (2 years)
end_date = datetime.today().date()
start_date = end_date - timedelta(days=2 * 365)
ticker = "ALO.PA"
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

# 2. Basic checks and flatten
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# 3. Technical indicators
data['Return'] = data['Close'].pct_change()
data['MA10'] = data['Close'].rolling(10).mean()
data['MA30'] = data['Close'].rolling(30).mean()
data['Momentum'] = data['Close'] - data['Close'].shift(10)
data['Volatility'] = data['Return'].rolling(10).std()

# Bollinger Bands
data['BB_upper'] = data['MA10'] + 2 * data['Volatility']
data['BB_lower'] = data['MA10'] - 2 * data['Volatility']

# OBV
obv = [0]
for i in range(1, len(data)):
    if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
        obv.append(obv[-1] + data['Volume'].iloc[i])
    elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
        obv.append(obv[-1] - data['Volume'].iloc[i])
    else:
        obv.append(obv[-1])
data['OBV'] = obv

# Stochastic Oscillator
low14 = data['Low'].rolling(window=14).min()
high14 = data['High'].rolling(window=14).max()
data['%K'] = 100 * ((data['Close'] - low14) / (high14 - low14))
data['%D'] = data['%K'].rolling(window=3).mean()

# Drop rows with NaNs
data.dropna(inplace=True)

# 4. Features and labels
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA30', 'Momentum',
            'Volatility', 'BB_upper', 'BB_lower', 'OBV', '%K', '%D']

X = data[features].iloc[:-1]
y_reg = data['Close'].shift(-1).iloc[:-1]
y_clf = (data['Close'].shift(-1) > data['Close']).astype(int).iloc[:-1]
latest_features = data[features].iloc[-1:]
last_close = data['Close'].iloc[-1]

# 5. Model training (no grid search for speed)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
reg_model.fit(X, y_reg)
clf_model.fit(X, y_clf)

# 6. Predictions
predicted_price = reg_model.predict(latest_features)[0]
predicted_direction = clf_model.predict(latest_features)[0]
reg_cv_score = -cross_val_score(reg_model, X, y_reg, cv=3, scoring='neg_mean_squared_error').mean()
clf_cv_score = cross_val_score(clf_model, X, y_clf, cv=3, scoring='accuracy').mean()

# 7. Summary
summary = f"""
🔍 Dernière clôture connue : €{last_close:.2f}
📈 Prix prédit pour demain : €{predicted_price:.2f}
🔁 Direction prévue : {'⬆️ UP' if predicted_direction == 1 else '⬇️ DOWN'}
📊 MSE régression (CV) : {reg_cv_score:.2f}
🎯 Précision classification (CV) : {clf_cv_score:.2%}
"""
print(summary)

# 8. Export results
output_dir = os.path.expanduser("~/Desktop/AI/prediction_outputs")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "latest_prediction.txt"), "w") as f:
    f.write(summary)

data.tail(1).assign(Predicted_Close=predicted_price,
                    Predicted_Direction='UP' if predicted_direction else 'DOWN')\
    .to_csv(os.path.join(output_dir, "latest_prediction.csv"), index=False)

# 9. Plot
plt.figure(figsize=(12, 6))
plt.plot(data['Close'].tail(60), label='Close Price')
plt.plot(data['MA10'].tail(60), label='MA10', linestyle='--')
plt.plot(data['BB_upper'].tail(60), label='BB Upper', linestyle='--', alpha=0.5)
plt.plot(data['BB_lower'].tail(60), label='BB Lower', linestyle='--', alpha=0.5)
plt.axhline(predicted_price, color='red', linestyle='--', label=f'Prévision: €{predicted_price:.2f}')
plt.title("Prix de clôture avec Bandes de Bollinger (60 derniers jours)")
plt.xlabel("Date")
plt.ylabel("Prix (€)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
