import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_preprocessing import preprocess_data, data_splitting
from feature_engineering import build_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def make_horizon_data(train_df, test_df, feature_cols, target_col, horizon):
    y_train_h = train_df[target_col].shift(-horizon).dropna()
    X_train_h = train_df.loc[y_train_h.index, feature_cols]

    y_test_h = test_df[target_col].shift(-horizon).dropna()
    X_test_h = test_df.loc[y_test_h.index, feature_cols]

    return X_train_h, y_train_h, X_test_h, y_test_h


def main():

    # Preprocessing data based on pre-processing file functions
    df = preprocess_data(
        path="air+quality/AirQualityUCI.xlsx",
        missing_strategy="interpolate",
        add_time_features_flag=True,
        normalize=False,
    )

    # Feature Engineering method call
    df = build_features(df)

    # Define class labels according to bounds specified in assignment spec
    df["CO_class"] = pd.cut(
        df["CO(GT)"], bins=[-np.inf, 1.5, 2.5, np.inf],
        labels=["low", "mid", "high"]
    )

    df = df.dropna().reset_index(drop=False)

    # Temporal data splitting 
    train_df, test_df = data_splitting(df)

    target_col = "CO_class"
    feature_cols = [c for c in df.columns if c not in ["DateTime", "CO(GT)", target_col]]

    horizons = [1, 6, 12, 24]

    for h in horizons:
        print("\n====================================")
        print(f"   RANDOM FOREST — HORIZON t+{h}")
        print("====================================")

        X_train_h, y_train_h, X_test_h, y_test_h = make_horizon_data(
            train_df, test_df, feature_cols, target_col, h
        )

        print("X_train_h:", X_train_h.shape, "X_test_h:", X_test_h.shape)

        # Train model accordingly
        rf = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=0
        )
        rf.fit(X_train_h, y_train_h)

        y_pred_h = rf.predict(X_test_h)

        # Calculate accuracy and f1 scores
        acc = accuracy_score(y_test_h, y_pred_h)
        f1 = f1_score(y_test_h, y_pred_h, average="weighted")

        # Print out evaluation metrics 
        print(f"RF Accuracy t+{h}: {acc:.3f}")
        print(f"RF F1 t+{h}: {f1:.3f}")

        naive_pred_h = test_df[target_col].loc[y_test_h.index]

        # Calculate accuracy and f1 scores
        naive_acc = accuracy_score(y_test_h, naive_pred_h)
        naive_f1 = f1_score(y_test_h, naive_pred_h, average="weighted")

        print(f"Naive Accuracy t+{h}: {naive_acc:.3f}")
        print(f"Naive F1 t+{h}: {naive_f1:.3f}")

        # Generate confusion matrix to observe correct vs. incorrect predictions
        cm = confusion_matrix(y_test_h, y_pred_h, labels=["low", "mid", "high"])
        print("\nConfusion Matrix:")
        print(cm)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["low", "mid", "high"],
            yticklabels=["low", "mid", "high"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Random Forest Confusion Matrix (t+{h})")
        plt.tight_layout()
        plt.savefig(f"rf_confusion_matrix_t+{h}.png", dpi=200)
        plt.close()

        # FEATURE IMPORTANCE BAR PLOT (to visualise which features are more important than others in prediction)
        importances = rf.feature_importances_
        idx = np.argsort(importances)[::-1][:15]

        top_features = np.array(feature_cols)[idx]
        top_scores = importances[idx]

        plt.figure(figsize=(10, 5))
        plt.bar(top_features, top_scores)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Random Forest Feature Importance (t+{h})")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
