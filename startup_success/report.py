from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
import typer

from startup_success.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
MODEL_PATH = MODELS_DIR / "balanced_random_forest.pkl"
PIPELINE_PATH = MODELS_DIR / "preprocessor_pipeline.pkl"
REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_PLOT_PATH = FIGURES_DIR / "confusion_matrix.png"


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], save_path: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.success(f"Saved confusion matrix to: {save_path}")


@app.command()
def main(features_path: Path = FEATURES_PATH):
    logger.info(f"Loading data from: {features_path}")
    df = pd.read_csv(features_path)

    if "success" not in df.columns:
        logger.error("'success' column is required for reporting.")
        raise typer.Exit(1)

    X = df.drop(columns=["success"])
    y_true = df["success"]

    logger.info("Loading model and pipeline...")
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)

    X_processed = pipeline.transform(X)
    y_proba = model.predict_proba(X_processed)[:, 1]

    # Threshold optimization (again on full dataset here for reporting purposes)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = [
        classification_report(
            y_true, (y_proba >= t).astype(int), output_dict=True, zero_division=0
        )["0"]["f1-score"]
        for t in thresholds
    ]
    best_index = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_index]

    y_pred = (y_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(
        y_true, y_pred, target_names=["Failure", "Success"], zero_division=0
    )
    pr_auc = average_precision_score(y_true, y_proba)

    logger.info("Generating classification report...")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Best Threshold: {best_threshold:.4f}\n")
        f.write(f"Precision-Recall AUC: {pr_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(clf_report)
    logger.success(f"Saved report to: {REPORT_PATH}")

    plot_confusion_matrix(cm, labels=["Failure", "Success"], save_path=CONFUSION_PLOT_PATH)


if __name__ == "__main__":
    app()
