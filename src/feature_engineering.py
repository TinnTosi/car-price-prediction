import pandas as pd
from pathlib import Path


# Path to dataset
DATA_PATH = Path("data/cars_cleaned.csv")
FEATURED_DATA_PATH = Path("data/cars_featured.csv")


# Load the cleaned dataset
df = pd.read_csv(DATA_PATH)


# Create car age
def _create_car_age(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["car_age"] = 2019 - df["year"]

    return df


# Create average mileage per year
def _create_mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["mileage_per_year"] = (
        df["mileage_kilometers"] / df["car_age"].replace(0, 1)
    )

    return df


# Convert engine volume from cm3 to liters
def _create_engine_volume_liters(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["engine_volume_liters"] = df["volume_cm3"] / 1000

    return df


# Connect all feature engineering operations
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df_featured = (
        df
        .pipe(_create_car_age)
        .pipe(_create_mileage_per_year)
        .pipe(_create_engine_volume_liters)
        .reset_index(drop=True)
    )

    return df_featured


# Apply feature engineering
df_featured = feature_engineering(df)


# Save the dataset with new features
df_featured.to_csv(FEATURED_DATA_PATH, index=False)
