# PPT Assets Guide

This guide collects the final project figures and tables that are most useful for slides.

All generated assets are saved in `ppt_assets/`.

## Recommended Slide Flow

### 1. Problem Setup
- Use: `ppt_assets/high_price_summary.png`
- Message: Listings priced at `$500+` are a small but high-variance tail of the data.
- Talk track: Most listings are below `$500`, so a single model across all listings is distorted by a small high-price segment.

### 2. Why Split the Task
- Use: `ppt_assets/high_price_room_type_distribution.png`
- Use: `ppt_assets/high_price_borough_distribution.png`
- Message: High-price listings are concentrated in `Entire home/apt` and heavily concentrated in `Manhattan`.
- Talk track: This shows high-price listings are not only fewer, but structurally different from the main sample.

### 3. Main Modeling Result
- Use: `ppt_assets/final_model_comparison.png`
- Message: The final setup is a two-part prediction strategy: a strong `<500` model and a separate high-price model.
- Talk track: Among tested approaches, random forest was the most stable model family. Restricting the main task to typical listings and segmenting by `room_type` gave the lowest test error, while the high-price model remains weaker because that segment is small and structurally different.

### 4. Performance Improvement
- Use: `ppt_assets/final_test_mse_comparison.png`
- Use: `ppt_assets/final_test_mae_comparison.png`
- Message: Splitting the task sharply improves prediction quality for the main price range.
- Talk track: The full-data global model has much higher error. Once the task is restricted to the dominant price range, performance improves substantially, and segmentation improves it further.

### 5. High-Price Analysis
- Use: `ppt_assets/top_high_price_neighborhoods.png`
- Message: The expensive tail is concentrated in a small number of neighborhoods.
- Talk track: This supports treating `$500+` listings as a separate analysis problem rather than forcing them into the main predictive task.

## Final Model Summary

- Main task: `price < 500`
- Main model: segmented `RandomForestRegressor` by `room_type`
- Secondary task: `price >= 500`
- Secondary model: global `RandomForestRegressor` for the high-price segment
- Parameters:
  - `n_estimators = 400`
  - `max_depth = 16`
  - `max_features = 'sqrt'`
  - `min_samples_leaf = 2`
  - `min_samples_split = 5`

## Final Numbers to Cite

- All listings, global random forest:
  - `Test MSE = 5499.530`
  - `Test MAE = 44.842`
  - `Test R2 = 0.492`

- Typical listings only (`<500`), global random forest:
  - `Test MSE = 3308.983`
  - `Test MAE = 38.426`
  - `Test R2 = 0.534`

- Typical listings only (`<500`), segmented random forest by `room_type`:
  - `Test MSE = 3198.021`
  - `Test MAE = 38.320`
  - `Test R2 = 0.555`

- High-price listings only (`>=500`), global random forest:
  - `Test MSE = 6786.837`
  - `Test MAE = 66.416`
  - `Test R2 = 0.133`

## Suggested Conclusion Slide Text

Random Forest was the strongest model family on this dataset. Error analysis showed that listings priced above `$500` are sparse, high-variance, and concentrated in specific room types and locations, which inflated the overall error of a single unified model. We therefore split the task into a main prediction problem for typical listings under `$500` and a separate high-price prediction problem. The best-performing main model was a segmented random forest by `room_type` on the `<500` subset, which reduced test MSE from `5499.53` on the all-listings global model to `3198.02`. The high-price model had weaker performance (`Test MSE = 6786.84`, `R2 = 0.13`), confirming that high-price listings likely require additional luxury-specific features or more data.
