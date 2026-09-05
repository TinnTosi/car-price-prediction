import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


# Target column
TARGET_COLUMN = "price_usd"


# Numerical features used for model training
NUMERIC_FEATURES = [
    "mileage_kilometers",
    "volume_cm3",
    "car_age",
    "mileage_per_year"
]


# Nominal categorical features
# These categories do not have a natural order
CATEGORICAL_FEATURES = [
    "make",
    "model",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment"
]


# Ordinal categorical features
# These categories have a natural order
ORDINAL_FEATURES = [
    "condition"
]


# Return all feature columns used by the model
def get_all_feature_columns() -> list[str]:

    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES


# Separate input features (X) from the target variable (y)
def split_features_and_target(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:

    X = df[get_all_feature_columns()].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y



# Build preprocessing pipeline for numerical features
def _build_numeric_transformer() -> Pipeline:

    numeric_transformer = Pipeline(
        steps=[
            # Fill missing numerical values with the median
            ("imputer", SimpleImputer(strategy="median")),

            # Standardize numerical values
            ("scaler", StandardScaler()),
        ]
    )

    return numeric_transformer



# Build preprocessing pipeline for nominal categorical features
def _build_categorical_transformer() -> Pipeline:

    categorical_transformer = Pipeline(
        steps=[
            # Fill missing categorical values with the most frequent value
            ("imputer", SimpleImputer(strategy="most_frequent")),

            # Convert categories into numerical binary columns
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return categorical_transformer


# Build preprocessing pipeline for ordinal categorical features
def _build_ordinal_transformer() -> Pipeline:

    ordinal_transformer = Pipeline(
        steps=[
            # Fill missing ordinal values with the most frequent value
            ("imputer", SimpleImputer(strategy="most_frequent")),

            # Encode categories according to their natural order
            ("encoder", OrdinalEncoder(
                categories=[
                    ["for parts", "with damage", "with mileage"]
                ]
            )),
        ]
    )

    return ordinal_transformer


# Combine numerical, nominal and ordinal preprocessing
def build_preprocessor() -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", _build_numeric_transformer(), NUMERIC_FEATURES),
            ("cat", _build_categorical_transformer(), CATEGORICAL_FEATURES),
            ("ord", _build_ordinal_transformer(), ORDINAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor
