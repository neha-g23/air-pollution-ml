import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define groups of columns, pollutants, sensor readings, meteorological variables, etc
POLLUTANT_COLS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
SENSOR_COLS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
               "PT08.S4(NO2)", "PT08.S5(O3)"]
METEO_COLS = ["T", "RH", "AH"]


# Helper function to load the dataset
def load_raw_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")

# Helper function to strip whitespace and drop unnamed columns from data
def clean_column_names(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    return df

# Helper function to merge date and time variables
def merge_datetime(df: pd.DataFrame):
    df = df.copy()

    if "DateTime" not in df.columns:
        df["DateTime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
            dayfirst=True
        )
        df.drop(["Date", "Time"], axis=1, inplace=True)

    df = df.dropna(subset=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def convert_decimal_separators(df: pd.DataFrame):
    """Convert numbers like '4,5' to float."""
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        series = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(series, errors="ignore")
    return df


# Dealing with missing values, replace -200 sentinal with NaN
def replace_sentinel_missing(df: pd.DataFrame, sentinel=-200):
    """Replace -200 with NaN."""
    df = df.copy()
    df.replace(sentinel, np.nan, inplace=True)
    return df


def handle_missing_values(df: pd.DataFrame, strategy="interpolate"):
    """Handle NaNs via none/drop/interpolate."""
    df = df.copy()

    # Do nothing
    if strategy == "none":
        return df

    # Drop all rows with NaN values
    if strategy == "drop":
        return df.dropna().reset_index(drop=True)

    # Time-based interpolation using DateTime as index
    # essentially fill missing values based on time stamps
    if strategy == "interpolate":
        df = df.set_index("DateTime").sort_index()
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].interpolate(method="time")
        df[num_cols] = df[num_cols].ffill().bfill()
        return df.reset_index()

    raise ValueError("Invalid missing strategy")

# Time features
def add_time_features(df: pd.DataFrame, add_year=True):
    """Add hour, weekday, month (+ year)."""
    df = df.copy()
    dt = pd.to_datetime(df["DateTime"])

    # add hour, weekday and month as temporal predictors 
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["month"] = dt.dt.month
    
    # add year to track annual effects
    if add_year:
        df["year"] = dt.dt.year
    return df


# Normalising
def _get_default_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Choose numeric cols except temporal ones."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num if c not in {"hour", "weekday", "month", "year"}]


def normalize_continuous_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    columns_to_scale: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """Z-score scaling using StandardScaler."""
    df_scaled = df.copy()

    if columns_to_scale is None:
        columns_to_scale = _get_default_numeric_columns(df_scaled)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df_scaled[columns_to_scale])

    df_scaled[columns_to_scale] = scaler.transform(df_scaled[columns_to_scale])
    return df_scaled, scaler, columns_to_scale


# Pipeline to perform all preprocessing methods
def preprocess_data(
    path: str,
    missing_strategy="interpolate",
    add_time_features_flag=True,
    normalize=False,
    scaler: Optional[StandardScaler] = None,
    columns_to_scale: Optional[List[str]] = None
):
    """Full preprocessing pipeline used across the project."""
    df = load_raw_data(path)
    df = clean_column_names(df)
    df = merge_datetime(df)
    df = convert_decimal_separators(df)
    df = replace_sentinel_missing(df)
    df = handle_missing_values(df, missing_strategy)

    if add_time_features_flag:
        df = add_time_features(df)

    if not normalize:
        return df

    df_scaled, scaler, cols = normalize_continuous_features(
        df, scaler=scaler, columns_to_scale=columns_to_scale
    )
    return df_scaled, scaler, cols

# Temporal data splitting 
def data_splitting(df):
    """Split the data into training (before 2005) and testing (2005 onwards) based on datetime."""
    df = df.copy()

    # Ensure DateTime is the index
    if df.index.name != 'DateTime':
        df = df.set_index('DateTime')

    # Define training and testing sets where 2004 data is used for training
    # and 2005 data used for testing
    train_df = df[df.index < '2005-01-01']
    test_df = df[df.index >= '2005-01-01']

    return train_df, test_df


# Check/validation
if __name__ == "__main__":
    df = preprocess_data(
        path="air+quality/AirQualityUCI.xlsx",
        missing_strategy="interpolate",
        normalize=False
    )
    train_df, test_df = data_splitting(df)
        # Count rows
    n_total = len(df)
    n_train = len(train_df)
    n_test = len(test_df)
    
    # Print sizes
    print(f"Total rows: {n_total}")
    print(f"Training rows: {n_train} ({n_train/n_total*100:.2f}%)")
    print(f"Testing rows: {n_test} ({n_test/n_total*100:.2f}%)")
    print(df.head())
    print(df.shape)

