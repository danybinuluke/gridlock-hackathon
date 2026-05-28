# Traffic Demand Prediction — Approach Document v2

## Problem
Predict traffic `demand` for 41,778 test samples given geohash location, time, road characteristics, weather, and temperature.

## Evaluation Metric
R² Score (coefficient of determination).

## Core Insight & Feature Engineering Strategy
The train data spans Day 48 (full day) and Day 49 (early morning). The test data spans the rest of Day 49. Because there is strong spatial-temporal autocorrelation, we construct historical profiles from Day 48 and map them to Day 49.

1. **Day 48 Geohash Profiles (Most Powerful Features):** 
   - Calculated `gh_hour_demand_mean`, `gh_demand_mean`, `gh_demand_std`, `gh_peak_demand`, `gh_offpeak_demand` using only Day 48 data. 
   - These profiles act as a direct historical lookup for the Day 49 test samples.
2. **Timestamp Features:** Hour, minute, weekday, month, time_slot (96 daily slots), peak_hour flag, night_hour flag.
3. **Cyclical Encoding:** Sin/cos transforms for hour (period=24), minute (period=60), weekday (period=7), time_slot (period=96).
4. **Geohash Prefix Aggregates:** Mean demand encoded for geohash prefixes (first 4 and 5 characters) to capture neighborhood-level trends.
5. **Interaction Features:** geohash×time_slot, geohash×hour, weather×hour, roadtype×hour, roadtype×lanes.
6. **Smoothed Target Encoding:** KFold out-of-fold target encoding applied to high-cardinality and interaction columns. Smoothing is applied to prevent overfitting on rare categories.

## Models
Three gradient boosting models trained with a **5-fold KFold** strategy. KFold is preferred over TimeSeriesSplit here because the train/test split is already perfectly temporal, and KFold maximizes the amount of data available to each training fold.
1. **CatBoost** (5000 iterations, lr=0.03)
2. **LightGBM** (5000 estimators, lr=0.03)
3. **XGBoost** (5000 estimators, lr=0.03)

## Ensemble
We use Scipy's `minimize` to find the optimal weighted blend of out-of-fold (OOF) predictions that maximizes the overall OOF R² score. This optimized blending reduces variance and significantly outperforms any individual model.

## Validation Strategy
5-Fold KFold cross-validation. Encoders, scalers, and imputation statistics are strictly fitted on the training split of each fold and applied to the validation split to prevent target leakage.

## Tools
Python, pandas, numpy, scikit-learn, LightGBM, CatBoost, XGBoost, Scipy, SHAP, pygeohash.
