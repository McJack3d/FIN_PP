import unittest
from src.models import train_model, evaluate_model
import pandas as pd
import numpy as np

class TestModels(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.X = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100)
        })
        self.y = np.random.random(100)
    
    def test_train_model(self):
        model, X_train, X_test, y_train, y_test, scaler = train_model(self.X, self.y)
        self.assertIsNotNone(model)
        self.assertEqual(len(X_train) + len(X_test), len(self.X))
    
    def test_evaluate_model(self):
        model, X_train, X_test, y_train, y_test, scaler = train_model(self.X, self.y)
        metrics = evaluate_model(model, X_test, y_test, self.X.columns)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)

if __name__ == '__main__':
    unittest.main()