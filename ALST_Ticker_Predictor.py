import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# --- 1. Download recent Alstom data (last 2 years) ---
end_date = datetime.today().date()
start_date = end_date - timedelta(days=4 * 365)
ticker = "ALO.PA"

data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

# On-Balance Volume (OBV)
obv = [0]
close = data['Close'].values
volume = data['Volume'].values

for i in range(1, len(close)):
    if close[i] > close[i - 1]:
        obv.append(obv[-1] + volume[i])
    elif close[i] < close[i - 1]:
        obv.append(obv[-1] - volume[i])
    else:
        obv.append(obv[-1])

data['OBV'] = obv

# Stochastic Oscillator
low14 = data['Low'].rolling(window=14).min()
high14 = data['High'].rolling(window=14).max()
data['%K'] = 100 * ((data['Close'] - low14) / (high14 - low14))
data['%D'] = data['%K'].rolling(window=3).mean()

# --- 2. Flatten columns if needed ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# --- 3. Feature engineering ---
data['Return'] = data['Close'].pct_change()
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA30'] = data['Close'].rolling(window=30).mean()
data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['EMA30'] = data['Close'].ewm(span=30, adjust=False).mean()
data['Momentum'] = data['Close'] - data['Close'].shift(10)
data['Volatility'] = data['Return'].rolling(window=10).std()

# Bollinger Bands
data['BB_upper'] = data['MA10'] + 2 * data['Volatility']
data['BB_lower'] = data['MA10'] - 2 * data['Volatility']

data.dropna(inplace=True)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))
data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

# --- 4. Define features and prepare training data ---
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA30', 'Momentum', 'Volatility']

X = data[features].iloc[:-1]  # all rows except the last one
y_reg = data['Close'].shift(-1).iloc[:-1]  # next day's price
y_clf = (data['Close'].shift(-1) > data['Close']).astype(int).iloc[:-1]  # next day's direction

# --- 5. Prepare the most recent row for prediction ---
latest_features = data[features].iloc[-1:]

# --- # --- 6. Train models ---
import os
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib'
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
latest_features_scaled = scaler.transform(latest_features)

reg_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
clf_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

from lightgbm import LGBMRegressor, LGBMClassifier

reg_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
clf_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# --- Hyperparameter Grid (peut être le même ou différent pour clf/reg) ---
param_grid_reg = { # Renommé pour clarté
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# --- Optimisation pour le Modèle de Régression ---
print("Optimisation du modèle de régression...")
# Utiliser TimeSeriesSplit pour la validation croisée temporelle
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5) # Exemple avec 5 splits

grid_search_reg = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid_reg,
    cv=tscv, # Utiliser TimeSeriesSplit
    scoring='neg_mean_squared_error',
    n_jobs=-1 # Utiliser tous les CPU disponibles
)
grid_search_reg.fit(X, y_reg) # Entraîne les modèles dans les plis et le meilleur modèle sur tout X

# Le meilleur modèle est déjà entraîné sur X, y_reg grâce à refit=True (défaut)
reg_model = grid_search_reg.best_estimator_
print(f"Meilleurs paramètres Régression: {grid_search_reg.best_params_}")
# PAS BESOIN de : reg_model.fit(X, y_reg)

# --- Optimisation pour le Modèle de Classification ---
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y_reg, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
# Vous pouvez utiliser la même grille ou une grille spécifique pour le classifieur
param_dist_clf = { # Utilisation de distributions pour RandomizedSearchCV si souhaité, ou grille fixe
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'criterion': ['gini', 'entropy'] # Paramètre spécifique au classifieur
}

print("Optimisation du modèle de classification...")
# n_iter contrôle le nombre de combinaisons testées (compromis vitesse/précision)
random_search_clf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist_clf, # Note: param_distributions pour RandomizedSearchCV
    n_iter=50, # Nombre d'itérations (à ajuster)
    cv=tscv, # Utiliser TimeSeriesSplit aussi ici
    scoring='accuracy', # Métrique de classification !
    random_state=42,
    n_jobs=-1
)
random_search_clf.fit(X, y_clf) # Utiliser y_clf !

# Le meilleur modèle est déjà entraîné sur X, y_clf
clf_model = random_search_clf.best_estimator_
print(f"Meilleurs paramètres Classification: {RandomForestClassifier(class_weight='balanced', random_state=42)}")
# PAS BESOIN de : clf_model.fit(X, y_clf)
# NE PAS ÉCRASER : # clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# NE PAS RÉENTRAINER : # clf_model.fit(X, y_clf)

# --- Prédictions (utilisent les modèles optimisés reg_model et clf_model) ---
predicted_price = reg_model.predict(latest_features)[0]
predicted_direction = clf_model.predict(latest_features)[0]

# --- 7. Save models ---
import joblib

joblib.dump(reg_model, 'reg_model.pkl')
joblib.dump(clf_model, 'clf_model.pkl')

# Load models
reg_model = joblib.load('reg_model.pkl')
clf_model = joblib.load('clf_model.pkl')

# --- 7. Predict for tomorrow ---
predicted_price = reg_model.predict(latest_features)[0]
predicted_direction = clf_model.predict(latest_features)[0]

# ---
from sklearn.model_selection import cross_val_score

reg_scores = cross_val_score(reg_model, X, y_reg, cv=5, scoring='neg_mean_squared_error')
clf_scores = cross_val_score(clf_model, X, y_clf, cv=5, scoring='accuracy')

print(f"Regression CV MSE: {-reg_scores.mean():.2f}")
print(f"Classification CV Accuracy: {clf_scores.mean():.2f}")

# --- 8. Output results ---
last_close = data['Close'].iloc[-1]
print(f"🔍 Last known close (Today): €{last_close:.2f}")
print(f"📈 Predicted next close (Tomorrow): €{predicted_price:.2f}")
print(f"🔁 Predicted direction: {'UP' if predicted_direction == 1 else 'DOWN'}")

# 8.1 Export results
output_dir = os.path.expanduser("~/Desktop/AI/prediction_outputs")
os.makedirs(output_dir, exist_ok=True)

summary = f"Last known close: €{last_close:.2f}\nPredicted next close: €{predicted_price:.2f}\nPredicted direction: {'UP' if predicted_direction == 1 else 'DOWN'}"

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
