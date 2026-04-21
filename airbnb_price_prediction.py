"""Airbnb price prediction models for NYC listings."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor

CV_FOLDS = 3


def build_preprocessor():
    """Create preprocessing steps shared by all models."""
    numeric_features = [
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    categorical_features = ["neighbourhood_group", "room_type"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_candidate(name, estimator, X_train, X_test, y_train, y_test):
    """Fit a candidate model and collect comparable metrics."""
    if isinstance(estimator, RandomizedSearchCV):
        estimator.fit(X_train, y_train)
        fitted_model = estimator.best_estimator_
        cv_mse = -estimator.best_score_
        notes = f"best params: {estimator.best_params_}"
    else:
        cv_scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        cv_mse = (-cv_scores).mean()
        fitted_model = estimator.fit(X_train, y_train)
        notes = ""

    train_pred = fitted_model.predict(X_train)
    test_pred = fitted_model.predict(X_test)

    return {
        "Model": name,
        "Train MSE": mean_squared_error(y_train, train_pred),
        "Test MSE": mean_squared_error(y_test, test_pred),
        "Test RMSE": mean_squared_error(y_test, test_pred) ** 0.5,
        "Test MAE": mean_absolute_error(y_test, test_pred),
        "Test R2": r2_score(y_test, test_pred),
        f"CV MSE ({CV_FOLDS}-fold)": cv_mse,
        "Notes": notes,
        "fitted_model": fitted_model,
    }


def main() -> None:
    """Train multiple regression models and compare their performance."""
    data_path = Path("AB_NYC_2019.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path.resolve()}")

    # Load data
    df = pd.read_csv(data_path)

    # Display basic dataset information
    print("Data preview:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    # Work on a copy so the raw frame remains intact for reference
    working_df = df.copy()

    # Handle missing values
    working_df["reviews_per_month"] = working_df["reviews_per_month"].fillna(0)
    working_df = working_df.dropna(subset=["neighbourhood_group", "room_type"])

    # Remove listings with non-positive prices and extreme outliers (top/bottom 1%)
    working_df = working_df[working_df["price"] > 0]
    lower_bound = working_df["price"].quantile(0.01)
    upper_bound = working_df["price"].quantile(0.99)
    working_df = working_df[(working_df["price"] >= lower_bound) & (working_df["price"] <= upper_bound)]

    # Select relevant features
    feature_columns = [
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "neighbourhood_group",
        "room_type",
    ]

    X = working_df[feature_columns]
    y = working_df["price"]

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    models = {
        "Linear Regression": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
        ),
        "Ridge (alpha=1.0)": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))]
        ),
        "Lasso (alpha=0.1)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Lasso(alpha=0.1, max_iter=10000)),
            ]
        ),
        "KNN (distance, k=15)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", KNeighborsRegressor(n_neighbors=15, weights="distance")),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeRegressor(max_depth=None, random_state=42)),
            ]
        ),
        "Random Forest Baseline": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200, random_state=42, n_jobs=1
                    ),
                ),
            ]
        ),
        "Gradient Boosting Baseline": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
        "Gradient Boosting + log(price)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=GradientBoostingRegressor(
                            random_state=42,
                            n_estimators=300,
                            learning_rate=0.05,
                            max_depth=3,
                            subsample=0.8,
                        ),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
    }

    tuned_models = {
        "Random Forest Tuned": RandomizedSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        RandomForestRegressor(random_state=42, n_jobs=1),
                    ),
                ]
            ),
            param_distributions={
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [12, 16, 20, None],
                "model__min_samples_leaf": [1, 2, 5],
                "model__max_features": ["sqrt", None],
            },
            n_iter=4,
            cv=CV_FOLDS,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=1,
        ),
        "Gradient Boosting Tuned": RandomizedSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", GradientBoostingRegressor(random_state=42)),
                ]
            ),
            param_distributions={
                "model__n_estimators": [150, 300, 500],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.8, 1.0],
                "model__min_samples_leaf": [1, 2, 5],
            },
            n_iter=4,
            cv=CV_FOLDS,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=1,
        ),
    }

    results = []

    for model_name, estimator in models.items():
        results.append(
            evaluate_candidate(
                model_name, estimator, X_train, X_test, y_train, y_test
            )
        )

    for model_name, estimator in tuned_models.items():
        results.append(
            evaluate_candidate(
                model_name, estimator, X_train, X_test, y_train, y_test
            )
        )

    trained_models = {row["Model"]: row.pop("fitted_model") for row in results}
    cv_col = f"CV MSE ({CV_FOLDS}-fold)"
    results_df = pd.DataFrame(results).sort_values(cv_col).reset_index(drop=True)
    print(f"\nModel performance (sorted by {cv_col}):")
    print(results_df.to_string(index=False, float_format="{:.2f}".format))

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest performing model by {cv_col}: {best_model_name}")

    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_predictions, alpha=0.5)
    max_price = max(y_test.max(), best_predictions.max())
    plt.plot([0, max_price], [0, max_price], color="red", linestyle="--", label="Ideal line")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted Prices ({best_model_name})")
    plt.legend()
    plt.tight_layout()
    plot_path = Path("pred_vs_actual.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {plot_path.resolve()}")


if __name__ == "__main__":
    main()
