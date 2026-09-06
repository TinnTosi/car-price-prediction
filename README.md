# Used Car Price Prediction

A machine learning regression project for predicting used car prices based on vehicle characteristics.

The project includes exploratory data analysis, data cleaning, feature engineering, preprocessing, model training, evaluation, and comparison of multiple regression algorithms.

## Dataset

The dataset contains information about used cars, including:

- make and model
- production year
- vehicle condition
- mileage
- fuel type
- engine volume
- color
- transmission
- drive unit
- vehicle segment
- price in USD

The target variable is **`price_usd`**.

## Project Structure

```text
car-price-prediction/
├── data/
│   └── cars.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
├── models/
│   └── car_price_model.joblib
├── README.md
└── requirements.txt
```

## Data Preparation

The data preparation process includes:

- standardizing column names and categorical values
- converting numerical columns
- removing duplicate records
- removing unrealistic price and mileage values
- handling missing values through preprocessing

## Feature Engineering

Two additional features were created:

- **car_age** – vehicle age based on the production year
- **mileage_per_year** – average mileage per year

## Preprocessing

Different preprocessing steps are applied depending on the feature type:

- numerical features: median imputation and standardization
- nominal categorical features: most-frequent imputation and one-hot encoding
- ordinal feature (`condition`): most-frequent imputation and ordinal encoding

The preprocessing steps and regression model are combined using a Scikit-learn `Pipeline`.

## Model Comparison

Four regression algorithms were compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The models were evaluated using MAE, MSE, RMSE, and R².

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Random Forest** | **1123.64** | **3096.38** | **0.867** |
| Decision Tree | 1450.41 | 3776.47 | 0.802 |
| Gradient Boosting | 1621.39 | 3379.38 | 0.841 |
| Linear Regression | 2078.69 | 4314.65 | 0.741 |

Random Forest achieved the best overall performance among the tested models, with the lowest MAE and RMSE and the highest R² score.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

The project files can then be run from the repository root.

## Conclusion

The project demonstrates a complete machine learning regression workflow, from exploratory data analysis and data preparation to model training and evaluation.

Among the tested algorithms, **Random Forest Regressor achieved the best performance**, with an R² score of approximately **0.87** and a mean absolute error of approximately **$1,124**.