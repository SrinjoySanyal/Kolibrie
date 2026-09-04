from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from basepredictor import BaseTimePredictor

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        
class Model2(BaseTimePredictor):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, feature_names=None):
        super().__init__(feature_names)
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        # Calculate prediction standard deviation
        return np.std([tree[0].predict(X_scaled) for tree in self.model.estimators_], axis=0)
        