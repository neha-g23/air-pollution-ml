
import pandas as pd
import numpy as np

from typing import List, Optional

POLLUTANTS = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

GT_TO_SENSOR = {
    "CO(GT)": "PT08.S1(CO)",
    "C6H6(GT)": "PT08.S2(NMHC)",
    "NOx(GT)": "PT08.S3(NOx)",
    "NO2(GT)": "PT08.S4(NO2)",
}

# Makes hour and month time frames cyclical to allow data to interpret time frames
# as continuous and repeating as opposed to linearly varying.
def create_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["sin_hour"] = np.sin(2*np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2*np.pi * df["hour"] / 24)
    
    df["sin_month"] = np.sin(2*np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2*np.pi * df["month"] / 12)
    
    return df

# Defines lag variables for 1, 2 and 24 hrs prior to specified instances of data (cols)
def create_lag_variables(df: pd.DataFrame, cols: List[str], lags: List[int] = [1, 2, 24]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

# Calculates the moving average of previous 6, 12, 24 hrs of data for each of the input columns of data provided
def create_moving_average_feature(df: pd.DataFrame, cols: List[str], window: List[int] = [6, 12, 24]) -> pd.DataFrame:
    df = df.copy()
    
    for col in cols:
        if col not in df.columns:
            continue
        for w in window:
            roll = df[col].rolling(window=w, min_periods=max(3, w // 2))
            df[f"{col}_roll_mean_{w}h"] = roll.mean()
            df[f"{col}_roll_std_{w}h"] = roll.std()
    return df

def build_features(df: pd.DataFrame, add_lag_variables: bool = True, add_moving_average: bool = True) -> pd.DataFrame:
    
    df_fe = df.copy()

    # Cyclical encodings for time
    df_fe = create_cyclical_time_features(df_fe)

    # Lags for pollutants and meteorology (only past info)
    if add_lag_variables:
        lag_cols = POLLUTANTS + ["T", "RH", "AH"]
        df_fe = create_lag_variables(df_fe, cols=lag_cols, lags=[1, 2, 24])

    # Moving average features for pollutants 
    if add_moving_average:
        roll_cols = POLLUTANTS
        df_fe = create_moving_average_feature(df_fe, cols=roll_cols, window=[6, 12, 24])

    return df_fe