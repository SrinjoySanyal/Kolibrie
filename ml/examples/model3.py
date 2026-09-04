from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os

from basepredictor import BaseTimePredictor

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        
class Model3(BaseTimePredictor):
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
        
