"""Airbnb price prediction models for NYC listings."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

CV_FOLDS = 3


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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso (alpha=0.1)": Lasso(alpha=0.1, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(max_depth=None, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    trained_models = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        cv_mse = (-cv_scores).mean()

        pipeline.fit(X_train, y_train)

        trained_models[model_name] = pipeline

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        results.append(
            {
                "Model": model_name,
                "Train MSE": train_mse,
                "Test MSE": test_mse,
                f"CV MSE ({CV_FOLDS}-fold)": cv_mse,
            }
        )

    results_df = pd.DataFrame(results).sort_values("Test MSE").reset_index(drop=True)
    print("\nModel performance (sorted by Test MSE):")
    print(results_df.to_string(index=False, float_format="{:.2f}".format))

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest performing model: {best_model_name}")

    # Optional visualization: predicted vs actual for best model
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
