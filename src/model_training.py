import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    split_features_and_target,
    build_preprocessor
)


# Paths to dataset and trained model
DATA_PATH = "data/cars_featured.csv"
MODEL_PATH = "models/car_price_model.joblib"


# Load the dataset
df = pd.read_csv(DATA_PATH)


# Separate input features (X) and target variable (y)
X, y = split_features_and_target(df)

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create the complete machine learning pipeline
print("Creating model pipeline...")

model = Pipeline(
    steps=[
        # Apply preprocessing to the input features
        ("preprocessor", build_preprocessor()),

        # Train a LinearRegression model
        (("regressor", LinearRegression())),
    ]
)


# Train the model using the training data
print("Training model...")

model.fit(X_train, y_train)


# Save the trained pipeline
print("Saving model...")

joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")