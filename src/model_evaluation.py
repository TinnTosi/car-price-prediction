import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import split_features_and_target


# Paths to dataset and trained model
DATA_PATH = "data/cars_featured.csv"
MODEL_PATH = "models/car_price_model.joblib"


# Load the dataset
df = pd.read_csv(DATA_PATH)


# Separate input features (X) and target variable (y)
X, y = split_features_and_target(df)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Load the trained model
loaded_model = joblib.load(MODEL_PATH)


# Make predictions on the test set
print("Making predictions...")

y_pred = loaded_model.predict(X_test)

print(y_pred[:10])


# Calculate regression metrics
print("Calculating regression metrics...")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


# Store evaluation metrics in a DataFrame
metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2]
})

print("\nRegression metrics:")
print(metrics)


# MAE → Average absolute difference between actual and predicted car prices
# MSE → Gives more weight to larger prediction errors
# RMSE → Average prediction error expressed in US dollars
# R² → Shows how much of the variation in car prices is explained by the model


# Create a table for prediction analysis
print("\nCreating prediction analysis table...")

prediction_analysis = pd.DataFrame({
    "actual_price_usd": y_test.values,
    "predicted_price_usd": y_pred
})


# Calculate prediction error
prediction_analysis["error_usd"] = (
    prediction_analysis["actual_price_usd"]
    - prediction_analysis["predicted_price_usd"]
)


# Calculate absolute prediction error
prediction_analysis["absolute_error_usd"] = (
    prediction_analysis["error_usd"].abs()
)


# Display random prediction examples
print("\nPrediction examples:")

print(
    prediction_analysis
    .sample(10, random_state=42)
)


# Display the largest prediction errors
print("\nLargest prediction errors:")

print(
    prediction_analysis
    .sort_values("absolute_error_usd", ascending=False)
    .head(10)
)