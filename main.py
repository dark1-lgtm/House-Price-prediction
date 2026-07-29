import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import joblib

df = pd.read_csv("house_prices_practice.csv")

print(df.head())

print(df.shape)

print(df.columns)

X = df.drop(["Id", "SalePrice"], axis=1)

Y = df["SalePrice"]

print(X.head())

print(Y.head())

scaler = StandardScaler()

X[X.columns] = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

print("X Train:", X_train.shape)
print("X Test :", X_test.shape)
print("Y Train:", Y_train.shape)
print("Y Test :", Y_test.shape)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, Y_train)

print("Model Trained Successfully")

model = LinearRegression()
model.fit(X_train, Y_train)
# Predict House Prices
Y_pred = model.predict(X_test)

print("Predicted Prices:")
print(Y_pred[:10])

print("\nActual Prices:")
print(Y_test.values[:10])

score = r2_score(Y_test, Y_pred)

print("R2 Score:", score)

# Save Model
joblib.dump(model, "house_model.pkl")

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

# Save Column Names
joblib.dump(X.columns.tolist(), "columns.pkl")

print("All Files Saved Successfully")