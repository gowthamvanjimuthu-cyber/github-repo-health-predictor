"""
Train Decision Tree and Random Forest classifiers for repository activity prediction.

Run: python train_model.py
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    MODELS_DIR,
    TARGET_COLUMN,
    FEATURE_COLUMNS_PATH,
)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run: python generate_dataset.py"
        )
    return pd.read_csv(DATASET_PATH)


def preprocess(df: pd.DataFrame):
    """Label encode target and split features."""
    X = df[FEATURE_COLUMNS].copy()
    y_raw = df[TARGET_COLUMN]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, label_encoder


def train_and_evaluate():
    df = load_dataset()
    print("=" * 60)
    print("GitHub Repository Active/Inactive - Model Training")
    print("=" * 60)
    print(f"\nDataset: {len(df)} samples")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"\nClass distribution:\n{df[TARGET_COLUMN].value_counts()}\n")

    X_train, X_test, y_train, y_test, label_encoder = preprocess(df)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        ),
    }

    best_model = None
    best_name = ""
    best_accuracy = 0.0

    for name, model in models.items():
        print("-" * 60)
        print(f"Training: {name}")
        print("-" * 60)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy Score: {accuracy:.4f}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(
            f"\nClassification Report:\n"
            f"{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}"
        )

        if hasattr(model, "feature_importances_"):
            importance = pd.Series(
                model.feature_importances_, index=FEATURE_COLUMNS
            ).sort_values(ascending=False)
            print(f"\nFeature Importance:\n{importance.to_string()}")

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    # Save best model (Random Forest preferred when tied)
    rf_model = models["Random Forest"]
    import numpy as np
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = np.concatenate([y_train, y_test])
    rf_model.fit(X_full, y_full)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_COLUMNS_PATH)

    print("\n" + "=" * 60)
    print(f"Best model: {best_name} (accuracy: {best_accuracy:.4f})")
    print(f"Saved Random Forest model to: {MODEL_PATH}")
    print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")
    print("=" * 60)

    return rf_model, label_encoder


if __name__ == "__main__":
    train_and_evaluate()
