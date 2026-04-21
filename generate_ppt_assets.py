from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("ppt_assets")
OUTPUT_DIR.mkdir(exist_ok=True)


FINAL_COMPARISON = pd.DataFrame(
    [
        {
            "Task": "All cleaned listings",
            "Model": "Global final random forest",
            "Sample Size": 48017,
            "Test MSE": 5499.529699,
            "Test RMSE": 74.158814,
            "Test MAE": 44.842373,
            "Test R2": 0.491705,
            "CV MSE (3-fold)": 5389.928008,
        },
        {
            "Task": "Typical listings only (<500)",
            "Model": "Typical global model (<500)",
            "Sample Size": 47256,
            "Test MSE": 3308.983214,
            "Test RMSE": 57.523762,
            "Test MAE": 38.425994,
            "Test R2": 0.534302,
            "CV MSE (3-fold)": 3344.583726,
        },
        {
            "Task": "Typical listings only (<500)",
            "Model": "Typical segmented model by room_type (<500)",
            "Sample Size": 47256,
            "Test MSE": 3198.021036,
            "Test RMSE": 56.551048,
            "Test MAE": 38.319598,
            "Test R2": 0.555099,
            "CV MSE (3-fold)": None,
        },
        {
            "Task": "High-price listings only (>=500)",
            "Model": "High-price global random forest",
            "Sample Size": 761,
            "Test MSE": 6786.837427,
            "Test RMSE": 82.382264,
            "Test MAE": 66.416002,
            "Test R2": 0.132731,
            "CV MSE (3-fold)": 7779.865856,
        },
    ]
)


HIGH_PRICE_SUMMARY = pd.Series(
    {
        "sample_size": 761,
        "mean_price": 606.629435,
        "median_price": 600.0,
        "std_price": 91.878635,
        "min_price": 500.0,
        "max_price": 799.0,
    }
)


HIGH_PRICE_ROOM_COUNTS = pd.Series(
    {
        "Entire home/apt": 665,
        "Private room": 92,
        "Shared room": 4,
    }
)


HIGH_PRICE_BOROUGH_COUNTS = pd.Series(
    {
        "Manhattan": 557,
        "Brooklyn": 176,
        "Queens": 20,
        "Bronx": 6,
        "Staten Island": 2,
    }
)


TOP_HIGH_PRICE_NEIGHBORHOODS = pd.Series(
    {
        "Midtown": 142,
        "Williamsburg": 56,
        "Hell's Kitchen": 50,
        "Upper West Side": 40,
        "Chelsea": 37,
        "West Village": 30,
        "East Village": 30,
        "Upper East Side": 30,
        "SoHo": 28,
        "Bedford-Stuyvesant": 22,
    }
)


def save_table_image(df: pd.DataFrame, title: str, filename: str) -> None:
    rows = len(df) + 1
    fig_height = max(2.8, 0.55 * rows)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=14)

    display_df = df.copy()
    for col in display_df.columns:
        if display_df[col].dtype.kind in "fc":
            display_df[col] = display_df[col].map(
                lambda x: "" if pd.isna(x) else f"{x:,.3f}"
            )

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f5d8a")
        elif row % 2 == 0:
            cell.set_facecolor("#eef3f8")
        else:
            cell.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_bar_chart(series: pd.Series, title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(kind="bar", color="#4c78a8", ax=ax)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    for idx, value in enumerate(series.values):
        ax.text(idx, value, f"{int(value)}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metric_chart(df: pd.DataFrame, metric: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [
        "All listings\nGlobal RF",
        "<500\nGlobal RF",
        "<500\nSegmented RF",
        ">=500\nHigh-price RF",
    ]
    ax.bar(labels, df[metric], color=["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"])
    ax.set_title(f"Comparison of {metric} Across Final Modeling Tasks", fontsize=15)
    ax.set_ylabel(metric)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    for idx, value in enumerate(df[metric]):
        ax.text(idx, value, f"{value:,.1f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FINAL_COMPARISON.to_csv(OUTPUT_DIR / "final_model_comparison.csv", index=False)
    HIGH_PRICE_SUMMARY.to_frame("value").to_csv(
        OUTPUT_DIR / "high_price_summary.csv", index=True
    )
    HIGH_PRICE_ROOM_COUNTS.to_frame("count").to_csv(
        OUTPUT_DIR / "high_price_room_type_counts.csv", index=True
    )
    HIGH_PRICE_BOROUGH_COUNTS.to_frame("count").to_csv(
        OUTPUT_DIR / "high_price_borough_counts.csv", index=True
    )
    TOP_HIGH_PRICE_NEIGHBORHOODS.to_frame("count").to_csv(
        OUTPUT_DIR / "top_high_price_neighborhoods.csv", index=True
    )

    save_table_image(
        FINAL_COMPARISON,
        "Final Model Comparison",
        "final_model_comparison.png",
    )
    save_table_image(
        HIGH_PRICE_SUMMARY.rename_axis("metric").reset_index(),
        "High-Price Listing Summary (>= $500)",
        "high_price_summary.png",
    )
    save_table_image(
        TOP_HIGH_PRICE_NEIGHBORHOODS.rename_axis("neighbourhood")
        .reset_index(name="count"),
        "Top High-Price Neighbourhoods",
        "top_high_price_neighborhoods.png",
    )

    save_metric_chart(FINAL_COMPARISON, "Test MSE", "final_test_mse_comparison.png")
    save_metric_chart(FINAL_COMPARISON, "Test MAE", "final_test_mae_comparison.png")
    save_bar_chart(
        HIGH_PRICE_ROOM_COUNTS,
        "High-Price Listings by Room Type",
        "Count",
        "high_price_room_type_distribution.png",
    )
    save_bar_chart(
        HIGH_PRICE_BOROUGH_COUNTS,
        "High-Price Listings by Borough",
        "Count",
        "high_price_borough_distribution.png",
    )


if __name__ == "__main__":
    main()
