"""Train baseline regressors on a cleaned Airbnb dataset."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("AB_NYC_2019.csv")  # Replace with cleaned dataset path if different
CV_FOLDS = 3  # Number of folds for cross-validation


def build_preprocessor(numeric_cols, categorical_cols):
    """Create preprocessing pipeline for numeric and categorical features."""
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
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def evaluate_model(
    name,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    results,
    cv_folds: int = CV_FOLDS,
):
    """Run cross-validation, fit on train split, and log all scores."""

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    cv_mse = (-cv_scores).mean()

    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    results.append(
        {
            "Model": name,
            "Train MSE": mean_squared_error(y_train, train_pred),
            "Test MSE": mean_squared_error(y_test, test_pred),
            f"CV MSE ({cv_folds}-fold)": cv_mse,
        }
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print(df.head())

    feature_cols = [
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

    X = df[feature_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_cols = [
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    categorical_cols = ["neighbourhood_group", "room_type"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models = {
        "Linear Regression": LinearRegression(),
        "KNN Regression": KNeighborsRegressor(n_neighbors=5),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(max_depth=None, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        evaluate_model(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
            results,
            cv_folds=CV_FOLDS,
        )

    results_df = pd.DataFrame(results)
    print("\nModel comparison (MSE):")
    print(results_df.to_string(index=False, float_format="{:.2f}".format))


if __name__ == "__main__":
    main()
