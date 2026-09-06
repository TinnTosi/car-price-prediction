import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import (
    split_features_and_target,
    build_preprocessor,
)


# Path to dataset
DATA_PATH = "data/cars_featured.csv"


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


# Define regression models for comparison
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}


# Store evaluation results
results = []


# Train and evaluate each regression model
for model_name, regressor in models.items():

    model = Pipeline(
        steps=[
            # Apply preprocessing
            ("preprocessor", build_preprocessor()),

            # Train the selected regression model
            ("regressor", regressor),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Save model results
    results.append({
        "model": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    })


# Create comparison table
results_df = pd.DataFrame(results)


# Sort models by MAE from best to worst
results_df = results_df.sort_values(
    by="mae",
    ascending=True
)


# Display model comparison
print("\nModel comparison:")
print(results_df)