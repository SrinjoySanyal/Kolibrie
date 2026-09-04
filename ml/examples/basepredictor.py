from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
import sys
import time
import psutil
import shutil

from mlschema import MLSchema

class BasePredictor:
    def __init__(self, feature_names=None):
        self.scaler = StandardScaler()
        self.feature_names = feature_names or ['temperature', 'humidity', 'occupancy']
        
    def train(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        start_time = time.time()
        self.model.fit(X_scaled, y)
        self.training_time = time.time() - start_time
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        process = psutil.Process(os.getpid())
        self.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        predictions = self.model.predict(X_scaled)
        # time the model needs to run on a sinle data point instead of multiple data points
        self.prediction_time = (time.time() - start_time) / len(X)

        print("execution time = " + str(self.prediction_time))
        
        # Capture CPU usage
        self.cpu_usage = psutil.cpu_percent(interval=0.1)
        
        return predictions
    
    def determine_goldilocks_batch(self, X):
        X_scaled = self.scaler.transform(X)
        min_time = np.inf
        min_batch = np.inf
        print("data cardinality: ", X.shape[0])
        for batch_size in range(1, X.shape[0]):
            prediction = np.array([])
            start_time = time.time()
            for start in range(0, X.shape[0], batch_size):
                if start + batch_size <= len(X):
                    batch_prediction = self.model.predict(X[start:(start + batch_size), :])
                    np.concatenate((prediction, batch_prediction), axis=0)
                else:
                    batch_prediction = self.model.predict(X[start:, :])
                    np.concatenate((prediction, batch_prediction), axis=0)
            prediction_runtime = time.time() - start_time
            if prediction_runtime < min_time:
                min_time = prediction_runtime
                min_batch = batch_size
                # print("prediction time = " + str(prediction_runtime) + " batch size = " + str(batch_size))
        print("goldilocks batch = " + str(min_batch))
        self.batch_goldilocks = min_batch
    
    def predict_with_batching(self, X):
        X_scaled = self.scaler.transform(X)
        prediction = np.array([])
        for start in range(0, X_scaled.shape[0], self.batch_goldilocks):
            if start + self.batch_goldilocks <= X_scaled.shape[0]:
                batch_prediction = self.model.predict(X_scaled[start:(start + self.batch_goldilocks), :])
                prediction = np.concatenate((prediction, batch_prediction), axis=0)
            else:
                batch_prediction = self.model.predict(X_scaled[start:, :])
                prediction = np.concatenate((prediction, batch_prediction), axis=0)
        return prediction
    
    def predict_proba(self, X):
        # Default implementation - override in subclasses if needed
        return None
    
    def get_performance_metrics(self):
        return {
            'training_time': getattr(self, 'training_time', 0),
            'prediction_time': getattr(self, 'prediction_time', 0),
            'memory_usage_mb': getattr(self, 'memory_usage', 0),
            'cpu_usage_percent': getattr(self, 'cpu_usage', 0),
            'goldilocks_batch': getattr(self, "batch_goldilocks")
        }
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def save_with_schema(self, filename, X_train, y_train, X_test, y_test):
        # Save model to pkl
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
        # Generate schema
        schema = MLSchema()
        
        # Define an evaluation function that captures performance metrics
        def eval_func(model, X_test, y_test):
            y_pred = model.predict(X_test)
            
            from sklearn.metrics import mean_squared_error, r2_score
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'training_time': model.get_performance_metrics().get('training_time', 0),
                'prediction_time': model.get_performance_metrics().get('prediction_time', 0),
                'memory_usage_mb': model.get_performance_metrics().get('memory_usage_mb', 0),
                'cpu_usage_percent': model.get_performance_metrics().get('cpu_usage_percent', 0),
                'goldilocks_batch': model.get_performance_metrics().get('batch_goldilocks', 0)
            }
            return metrics
        
        # Generate schema
        schema.convert_model(
            self,
            X_train, y_train,
            X_test, y_test,
            feature_names=self.feature_names,
            cpu_time_used=self.get_performance_metrics().get('training_time', 0),
            model_uri=f"http://example.org/models/{os.path.basename(filename)}",
            evaluation_function=eval_func
        )
        
        # Save schema to file
        schema_filename = filename.replace('.pkl', '.ttl')
        with open(schema_filename, 'w') as f:
            f.write(schema.serialize(format='turtle'))
        
        return schema_filename
    
    def evaluate(self, X_test, y_test):
        """Calculate evaluation metrics and store them"""
        from sklearn.metrics import mean_squared_error, r2_score
        
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.evaluation_metrics = {
            'mse': mse,
            'r2': r2
        }
        
        return self.evaluation_metrics
    
    def get(self, attribute_name):
        """Helper method to get attributes safely"""
        return getattr(self, attribute_name, None)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
class BaseTimePredictor(BasePredictor):
    def predict_with_batching(self, X):
        X_scaled = self.scaler.transform(X)
        prediction = np.array([])
        for start in range(0, X_scaled.shape[0], self.batch_goldilocks):
            if start + self.batch_goldilocks <= X_scaled.shape[0]:
                batch_prediction = self.model.predict(X_scaled[start:(start + self.batch_goldilocks), :])
                prediction = np.concatenate((prediction, batch_prediction), axis=0)
            else:
                batch_prediction = self.model.predict(X_scaled[start:, :])
                prediction = np.concatenate((prediction, batch_prediction), axis=0)
        
        prediction[prediction < 250] = 1
        prediction[prediction >= 250] = 0
        return prediction
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        process = psutil.Process(os.getpid())
        self.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        predictions = self.model.predict(X_scaled)
        # time the model needs to run on a sinle data point instead of multiple data points
        self.prediction_time = (time.time() - start_time) / len(X)

        print("execution time = " + str(self.prediction_time))
        
        # Capture CPU usage
        self.cpu_usage = psutil.cpu_percent(interval=0.1)
        
        predictions[predictions < 250] = 1
        predictions[predictions >= 250] = 0
        return predictions