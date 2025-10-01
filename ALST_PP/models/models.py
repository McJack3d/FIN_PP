"""
Model creation and selection for stock price prediction
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel

def create_model(model_type='histgb', random_state=42):
    """Create a model based on specified type"""
    if model_type == 'rf':
        return RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)
    elif model_type == 'stacked':
        return create_stacked_model(random_state)
    elif model_type == 'ensemble':
        return create_ensemble(random_state)
    else:  # Default is histgb
        return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=300, random_state=random_state)

def create_ensemble(random_state=42):
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

def create_stacked_model(random_state=42):
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

def select_features(X, y, top_n=30):
    """Select most important features using Random Forest"""
    if len(X) <= 100:  # Not enough data
        return X.columns.tolist()[:min(top_n, len(X.columns))]
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return importances.nlargest(min(top_n, len(importances))).index.tolist()
