# startup_success/modeling/train.py

from pathlib import Path

import pandas as pd
import numpy as np

from loguru import logger
import typer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, classification_report,
    confusion_matrix, average_precision_score
)
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTETomek

from startup_success.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"

@app.command()
def main(features_path: Path = FEATURES_PATH):
    logger.info(f"Loading training data from: {features_path}")
    df = pd.read_csv(features_path)

    X = df.drop(columns=["success"])
    y = df["success"]

    categorical_features = ["country_code", "region"]
    numeric_features = [
        "funding_total_usd", "funding_rounds", "founded_year",
        "first_funding_year", "last_funding_year",
        "days_to_first_funding", "funding_duration"
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], remainder="passthrough")

    pipeline = Pipeline([("preprocessor", preprocessor)])

    logger.info("Splitting and transforming dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    logger.info("Applying SMOTE + Tomek resampling...")
    sampler = SMOTETomek(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)

    logger.info("Training Balanced Random Forest...")
    model = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_resampled, y_resampled)

    logger.info("Optimizing decision threshold...")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    f1_scores = [
        classification_report(y_test, (y_proba >= t).astype(int),
                              output_dict=True, zero_division=0)['0']['f1-score']
        for t in thresholds
    ]
    best_index = int(np.argmax(f1_scores))
    optimal_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    y_pred = (y_proba >= optimal_threshold).astype(int)
    pr_auc = average_precision_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Failure", "Success"], zero_division=0)

    logger.success("=== Model Performance Summary ===")
    logger.info(f"Optimized Threshold: {optimal_threshold:.4f}")
    logger.info(f"Best Class 0 F1-Score: {best_f1:.4f}")
    logger.info(f"Precision-Recall AUC: {pr_auc:.4f}")
    logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
    logger.info(f"\nClassification Report:\n{report}")


if __name__ == "__main__":
    app()
