#
# Copyright © 2024 Volodymyr Kadzhaia
# Copyright © 2024 Pieter Bonte
# KU Leuven — Stream Intelligence Lab, Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
#

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
import time
import psutil
from basepredictor import BasePredictor

class LinearRegressionPredictor(BasePredictor):
    def __init__(self, fit_intercept=True, normalize=None, feature_names=None):
        super().__init__(feature_names)

        # In scikit-learn 1.0+, normalize parameter was removed
        # Check scikit-learn version
        import sklearn
        from packaging import version

        try:
            if version.parse(sklearn.__version__) >= version.parse('1.0.0'):
                # For scikit-learn 1.0+
                self.model = LinearRegression(fit_intercept=fit_intercept)
                if normalize:
                    print("Warning: 'normalize' parameter is deprecated in scikit-learn 1.0+. Using StandardScaler instead.")
            else:
                # For scikit-learn < 1.0
                self.model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
        except Exception as e:
            print(f"Error initializing LinearRegression: {e}")
            # Fallback to simplest constructor
            self.model = LinearRegression()

    def predict_proba(self, X):
        # Linear regression doesn't have built-in uncertainty estimation
        # Return a simple constant uncertainty value
        X_scaled = self.scaler.transform(X)
        return np.ones(X_scaled.shape[0]) * 0.5  # Constant uncertainty

class RandomForestPredictor(BasePredictor):
    def __init__(self, n_estimators=100, max_depth=10, random_state=42, feature_names=None):
        super().__init__(feature_names)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = []
        for tree in self.model.estimators_:
            predictions.append(tree.predict(X_scaled))
        return np.std(predictions, axis=0)

class GradientBoostingPredictor(BasePredictor):
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

# Generate training data
np.random.seed(42)
n_samples = 1000

temperature = np.random.normal(22, 5, n_samples)
humidity = np.random.normal(50, 15, n_samples)
occupancy = np.random.randint(0, 20, n_samples)

# Create target variable with some noise
future_temp = (
    temperature * 0.7 +
    (humidity - 50) * 0.02 +
    occupancy * 0.1 +
    np.random.normal(0, 1, n_samples)
)

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X = np.column_stack([temperature, humidity, occupancy])
y = future_temp
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save models
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# RandomForest model
rf_model = RandomForestPredictor()
rf_model.train(X_train, y_train)
rf_model.predict(X_test)  # Run once to get performance metrics
rf_schema_file = rf_model.save_with_schema(os.path.join(models_dir, "rf_temperature_predictor.pkl"),
                                         X_train, y_train, X_test, y_test)

# GradientBoosting model
gb_model = GradientBoostingPredictor()
gb_model.train(X_train, y_train)
gb_model.predict(X_test)  # Run once to get performance metrics
gb_schema_file = gb_model.save_with_schema(os.path.join(models_dir, "gb_temperature_predictor.pkl"),
                                         X_train, y_train, X_test, y_test)

# Linear Regression model
lr_model = LinearRegressionPredictor()
lr_model.train(X_train, y_train)
lr_model.predict(X_test)  # Run once to get performance metrics
lr_schema_file = lr_model.save_with_schema(os.path.join(models_dir, "lr_temperature_predictor.pkl"),
                                         X_train, y_train, X_test, y_test)

print(f"RandomForest model saved to {os.path.join(models_dir, 'rf_temperature_predictor.pkl')}")
print(f"RandomForest schema saved to {rf_schema_file}")
print(f"GradientBoosting model saved to {os.path.join(models_dir, 'gb_temperature_predictor.pkl')}")
print(f"GradientBoosting schema saved to {gb_schema_file}")
print(f"LinearRegression model saved to {os.path.join(models_dir, 'lr_temperature_predictor.pkl')}")
print(f"LinearRegression schema saved to {lr_schema_file}")

print("\nPerformance Comparison:")
rf_metrics = rf_model.get_performance_metrics()
gb_metrics = gb_model.get_performance_metrics()
lr_metrics = lr_model.get_performance_metrics()

print("\nRandomForest Model:")
for key, value in rf_metrics.items():
    print(f"  {key}: {value}")

print("\nGradientBoosting Model:")
for key, value in gb_metrics.items():
    print(f"  {key}: {value}")

print("\nLinearRegression Model:")
for key, value in lr_metrics.items():
    print(f"  {key}: {value}")
