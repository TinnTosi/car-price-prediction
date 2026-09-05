import pandas as pd
from pathlib import Path


# Path to dataset
DATA_PATH = Path("data/cars.csv")
CLEANED_DATA_PATH = Path("data/cars_cleaned.csv")


# Load the dataset
df = pd.read_csv(DATA_PATH)


# Standardize column names
def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    new_columns = []

    for col in df.columns:
        clean_col = col.strip().lower()

        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")

        new_columns.append(clean_col)

    df.columns = new_columns

    df = df.rename(columns={"priceusd": "price_usd"})

    return df

# Converting numeric columns
def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    numeric_columns = [
        "price_usd",
        "year",
        "mileage_kilometers",
        "volume_cm3"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# Standardizing categorical columns
def _standardize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for col in categorical_columns:
        df[col] = df[col].str.strip().str.lower()

    return df

# Removing duplicates
def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df = df.drop_duplicates()

    return df



# Remove unrealistic numerical values
def _remove_unrealistic_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df[df["price_usd"] >= 500]
    df = df[df["mileage_kilometers"] <= 1_000_000]

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = (
        df
        .pipe(_standardize_column_names)
        .pipe(_convert_numeric_columns)
        .pipe(_standardize_categorical_columns)
        .pipe(_remove_duplicates)
        .pipe(_remove_unrealistic_values)
        .reset_index(drop=True)

    )

    return df_clean

df_cleaned = clean(df)
df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
