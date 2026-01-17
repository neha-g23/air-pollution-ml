import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from feature_engineering import build_features
from data_preprocessing import preprocess_data, data_splitting
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import warnings
from IPython.display import display
# Pollutants to predict
pollutants = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
# Time horizons to predict (in hours)
horizons = [1, 6, 12, 24]
# Load and preprocess
df = preprocess_data(
    path="air+quality/AirQualityUCI.xlsx", 
    missing_strategy="interpolate",
    normalize=False
)

# Temporal train-test split
train_df, test_df = data_splitting(df)

# Apply feature engineering (lags, moving averages, cyclical time)
train_df_fe = build_features(train_df)
test_df_fe = build_features(test_df)

visuals = {}
for pollutant in pollutants:
    for h in horizons:
        # Set target: value of pollutant h hours ahead
        train_df_fe["target"] = train_df_fe[pollutant].shift(-h)
        test_df_fe["target"] = test_df_fe[pollutant].shift(-h)
        # Drop rows with NaNs caused by lag, rolling, or shifting
        train_data = train_df_fe.dropna()
        test_data = test_df_fe.dropna()
        # Define features
        feature_cols = [
            col for col in train_data.columns
            if col not in ["target", pollutant]
        ]
        X_train = train_data[feature_cols]
        y_train = train_data["target"]
        X_test = test_data[feature_cols]
        y_test = test_data["target"]
        # Model setup
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        # Train model
        model.fit(X_train, y_train)
        # Predict on test set
        y_pred = model.predict(X_test)
        # Store model and relevant data for visualization
        visuals[(pollutant, h)] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred
        }
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        # Naive prediction
        naive_pred = test_data[pollutant].values
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    

        print(f"{pollutant} | {h}h — RMSE: {rmse:.3f} | R²: {r2:.3f} | Naive RMSE: {naive_rmse:.3f}")

import matplotlib.pyplot as plt

def plot_predictions_grid(visuals, model_name="Model"):
    pollutants = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
    horizons = [1, 6, 12, 24]

    for pollutant in pollutants:
        fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
        axs = axs.flatten()

        for idx, h in enumerate(horizons):
            data = visuals.get((pollutant, h))
            if data:
                y_test = data["y_test"]
                y_pred = data["y_pred"]

                axs[idx].scatter(y_test.index, y_test, label="Actual", alpha=0.7)
                axs[idx].scatter(y_test.index, y_pred, label="Predicted", alpha=0.7)
                axs[idx].set_title(f"{pollutant} | {h}h ahead")
                axs[idx].set_xlabel("Time")
                axs[idx].set_ylabel("Concentration")
                axs[idx].legend()

        plt.suptitle(f"{model_name}: {pollutant} — Actual vs Predicted", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

plot_predictions_grid(visuals, model_name="Linear Regression")
