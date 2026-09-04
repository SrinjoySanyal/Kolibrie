import os
from model1 import Model1
from model2 import Model2
from model3 import Model3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

csv_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "kolibrie",
        "examples",
        "smart_manufacturing_data.csv",
    )
)

df = pd.read_csv(csv_path)

feature_names = [
    "humidity",
    "temperature",
    "energy_consumption",
    "vibration",
]

target_name = "predicted_remaining_life"

X = df[feature_names].to_numpy()[1000:]
y = np.array(df[target_name].to_numpy()[1000:], copy=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

model = Model1(feature_names=feature_names)
model.train(X_train, y_train)
predictions = model.predict(X_test) # used to get the prediction times
model.determine_goldilocks_batch(X_test) 
model.save_with_schema(os.path.join(models_dir, "model1.pkl"), 
                                         X_train, y_train, X_test, y_test)

model = Model2(feature_names=feature_names)
model.train(X_train, y_train)
predictions = model.predict(X_test) # used to get the prediction times
model.determine_goldilocks_batch(X_test)
model.save_with_schema(os.path.join(models_dir, "model2.pkl"), 
                                         X_train, y_train, X_test, y_test)

model = Model3(feature_names=feature_names)
model.train(X_train, y_train)
predictions = model.predict(X_test) # used to get the prediction times
model.determine_goldilocks_batch(X_test)
model.save_with_schema(os.path.join(models_dir, "model3.pkl"), 
                                         X_train, y_train, X_test, y_test)