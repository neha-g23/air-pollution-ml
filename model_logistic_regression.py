"""
model_logistic_regression.py
Train + evaluate multiclass Logistic Regression models
for multiple forecast horizons (t+1, t+6, t+12, t+24)
using the full preprocessing + feature engineering pipeline,
and compare against the naive persistence baseline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from data_preprocessing import preprocess_data
from feature_engineering import build_features

# Helper: build horizon-specific datasets
def make_horizon_data(df_train, df_test, feature_cols, target_col, horizon):
    """
    For a given horizon h, construct X_train, y_train, X_test, y_test such that:
      - Features are at time t
      - Target is class at time t+h
    """
    # TRAIN
    y_train_h = df_train[target_col].shift(-horizon).dropna()
    X_train_h = df_train.loc[y_train_h.index, feature_cols]

    # TEST
    y_test_h = df_test[target_col].shift(-horizon).dropna()
    X_test_h = df_test.loc[y_test_h.index, feature_cols]

    return X_train_h, y_train_h, X_test_h, y_test_h


def main():

    RAW_PATH = "./air+quality/AirQualityUCI.xlsx"

    df = preprocess_data(
        path=RAW_PATH,
        missing_strategy="interpolate",
        add_time_features_flag=True,
        normalize=False,
    )

    print("After preprocessing:", df.shape)

    df = build_features(df)
    print("After feature engineering:", df.shape)

    df = df.dropna().reset_index(drop=False)

    # CREATE CLASSIFICATION TARGET — CO Levels (spec thresholds)
    df["CO_Level"] = pd.cut(
        df["CO(GT)"],
        bins=[-np.inf, 1.5, 2.5, np.inf],
        labels=["Low", "Medium", "High"]
    )

    print("\nClass distribution:")
    print(df["CO_Level"].value_counts())

    df = df.sort_values("DateTime").reset_index(drop=True)

    train_df = df[df["DateTime"] < "2005-01-01"]
    test_df  = df[df["DateTime"] >= "2005-01-01"]

    print("\nTrain size:", train_df.shape)
    print("Test size :", test_df.shape)

    target_col = "CO_Level"
    feature_cols = [
        c for c in df.columns
        if c not in ["DateTime", target_col]
    ]

    print("Num features:", len(feature_cols))


    horizons = [1, 6, 12, 24]

    for h in horizons:
        print("\n====================================")
        print(f"LOGISTIC REGRESSION — HORIZON t+{h}")
        print("====================================")

        # Build horizon-specific data
        X_train_h, y_train_h, X_test_h, y_test_h = make_horizon_data(
            train_df, test_df, feature_cols, target_col, h
        )

        print("X_train_h:", X_train_h.shape, "X_test_h:", X_test_h.shape)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_h)
        X_test_scaled  = scaler.transform(X_test_h)

        # Train logistic regression for this horizon
        log_reg = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=500,
        )
        log_reg.fit(X_train_scaled, y_train_h)

        # Evaluation (model)
        y_pred_h = log_reg.predict(X_test_scaled)

        acc = accuracy_score(y_test_h, y_pred_h)
        macro_f1 = f1_score(y_test_h, y_pred_h, average="macro")
        weighted_f1 = f1_score(y_test_h, y_pred_h, average="weighted")
        cm = confusion_matrix(y_test_h, y_pred_h, labels=["Low", "Medium", "High"])

        print("Accuracy      :", acc)
        print("Macro F1      :", macro_f1)
        print("Weighted F1   :", weighted_f1)
        print("\nClassification Report:")
        print(classification_report(y_test_h, y_pred_h))

        # Save confusion matrix for EVERY horizon as a PNG
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix — Logistic Regression (t+{h})")
        plt.tight_layout()
        plt.savefig(f"logreg_confusion_matrix_t+{h}.png", dpi=200)
        plt.close()

        # Naive baseline for this horizon
        # True label is CO_Level at t+h (y_test_h).
        # Naive prediction for t+h is simply CO_Level at time t.
        naive_pred_h = test_df[target_col].loc[y_test_h.index]

        naive_acc = accuracy_score(y_test_h, naive_pred_h)
        naive_f1  = f1_score(y_test_h, naive_pred_h, average="weighted")

        print("\nNaive baseline (persistence) for t+{}:".format(h))
        print("Naive ACC:", naive_acc)
        print("Naive F1 :", naive_f1)

    print("\nDone.")


if __name__ == "__main__":
    main()
