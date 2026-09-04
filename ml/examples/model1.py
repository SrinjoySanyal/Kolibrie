from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from basepredictor import BaseTimePredictor

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        
class Model1(BaseTimePredictor):
    def __init__(self, n_estimators=100, max_depth=10, random_state=42, feature_names=None):
        super().__init__(feature_names)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
