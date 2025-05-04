# startup_success/modeling/train.py

from pathlib import Path

from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
import joblib
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from startup_success.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
MODEL_PATH = MODELS_DIR / "balanced_random_forest.pkl"
PIPELINE_PATH = MODELS_DIR / "preprocessor_pipeline.pkl"
REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_PLOT_PATH = FIGURES_DIR / "confusion_matrix.png"
PR_CURVE_PATH = FIGURES_DIR / "precision_recall_curve.png"
F1_CURVE_PATH = FIGURES_DIR / "threshold_vs_f1.png"
ROC_CURVE_PATH = FIGURES_DIR / "roc_curve.png"


def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.success(f"Saved confusion matrix to: {save_path}")


def plot_precision_recall_curve(precision, recall, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="PR Curve", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.success(f"Saved Precision-Recall curve to: {save_path}")


def plot_threshold_vs_f1(thresholds, f1_scores, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, marker="o", label="F1-Score")
    plt.axvline(
        x=thresholds[np.argmax(f1_scores)], color="red", linestyle="--", label="Best Threshold"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Class 0 F1-Score")
    plt.title("F1-Score vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.success(f"Saved F1-score vs Threshold plot: {save_path}")


def plot_roc_curve(y_true, y_proba, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.success(f"Saved ROC curve: {save_path}")


@app.command()
def main(features_path: Path = FEATURES_PATH):
    logger.info(f"Loading training data from: {features_path}")
    df = pd.read_csv(features_path)

    X = df.drop(columns=["success"])
    y = df["success"]

    categorical_features = ["country_code", "region"]
    numeric_features = [
        "funding_total_usd",
        "funding_rounds",
        "founded_year",
        "first_funding_year",
        "last_funding_year",
        "days_to_first_funding",
        "funding_duration",
    ]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

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
        n_estimators=500, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=42
    )
    model.fit(X_resampled, y_resampled)

    logger.info("Optimizing decision threshold...")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    f1_scores = [
        classification_report(
            y_test, (y_proba >= t).astype(int), output_dict=True, zero_division=0
        )["0"]["f1-score"]
        for t in thresholds
    ]
    best_index = int(np.argmax(f1_scores))
    optimal_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    y_pred = (y_proba >= optimal_threshold).astype(int)
    pr_auc = average_precision_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Failure", "Success"], zero_division=0
    )

    # === Reporting and Plotting ===
    logger.success("=== Model Performance Summary ===")
    logger.info(f"Optimized Threshold: {optimal_threshold:.4f}")
    logger.info(f"Best Class 0 F1-Score: {best_f1:.4f}")
    logger.info(f"Precision-Recall AUC: {pr_auc:.4f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Best Threshold: {optimal_threshold:.4f}\n")
        f.write(f"Precision-Recall AUC: {pr_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    logger.success(f"Saved classification report to: {REPORT_PATH}")

    # Generate Plots: Confusion Matrix, Precision Recall, Threshold vs F1, ROC
    plot_confusion_matrix(
        conf_matrix, labels=["Failure", "Success"], save_path=CONFUSION_PLOT_PATH
    )
    plot_precision_recall_curve(precision, recall, save_path=PR_CURVE_PATH)
    plot_threshold_vs_f1(thresholds, f1_scores, save_path=F1_CURVE_PATH)
    plot_roc_curve(y_test, y_proba, save_path=ROC_CURVE_PATH)

    # === Save Model and Pipeline ===
    logger.info(f"Saving model to: {MODEL_PATH}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.success("Model saved.")

    logger.info(f"Saving preprocessing pipeline to: {PIPELINE_PATH}")
    joblib.dump(pipeline, PIPELINE_PATH)
    logger.success("Preprocessing pipeline saved.")


if __name__ == "__main__":
    app()
