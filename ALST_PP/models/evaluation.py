"""
Model evaluation and metrics for stock price prediction
"""
import numpy as np
import pandas as pd
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV

from .models import create_model, select_features

def evaluate_models(X, y_ret, model_type='ensemble', calibrate=True, random_state=42):
    """
    Evaluate regression and classification models using time series cross-validation
    """
    # Feature selection
    top_features = select_features(X, y_ret)
    X_reduced = X[top_features]
    
    # Initialize models
    reg_model = RandomForestRegressor(random_state=random_state, n_estimators=500, max_depth=None, n_jobs=-1)
    clf = create_model(model_type, random_state)
    if calibrate:
        clf = CalibratedClassifierCV(clf, cv=3, method='isotonic')
    
    # Time-aware CV for metrics
    n_splits = max(3, min(8, len(X) // 80))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)
    
    # Initialize metrics arrays
    maes, rmses, accs, precs, recs, f1s, aucs = [], [], [], [], [], [], []
    
    # Cross-validation
    for tr, te in tscv.split(X_reduced):
        X_tr, X_te = X_reduced.iloc[tr], X_reduced.iloc[te]
        yr_tr, yr_te = y_ret.iloc[tr], y_ret.iloc[te]
        
        # Regression evaluation
        reg_model.fit(X_tr, yr_tr)
        pred_ret = reg_model.predict(X_te)
        maes.append(mean_absolute_error(yr_te, pred_ret))
        rmses.append(np.sqrt(mean_squared_error(yr_te, pred_ret)))
        
        # Classification evaluation
        y_tr_bin = (yr_tr > 0).astype(int)
        y_te_bin = (yr_te > 0).astype(int)
        
        if calibrate:
            clf_model = CalibratedClassifierCV(create_model(model_type, random_state), cv=3, method='isotonic')
        else:
            clf_model = create_model(model_type, random_state)
            
        clf_model.fit(X_tr, y_tr_bin)
        pred_lbl = clf_model.predict(X_te)
        
        # Classification metrics
        accs.append(accuracy_score(y_te_bin, pred_lbl))
        precs.append(precision_score(y_te_bin, pred_lbl, zero_division=0))
        recs.append(recall_score(y_te_bin, pred_lbl, zero_division=0))
        f1s.append(f1_score(y_te_bin, pred_lbl, zero_division=0))
        
        try:
            proba = clf_model.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te_bin, proba))
        except Exception:
            aucs.append(float('nan'))
    
    # Train final models on full dataset
    reg_model.fit(X_reduced, y_ret)
    y_bin = (y_ret > 0).astype(int)
    clf.fit(X_reduced, y_bin)
    
    # Compile metrics
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
    }
    
    return {
        'reg_model': reg_model,
        'clf_model': clf,
        'metrics': metrics,
        'top_features': top_features
    }
