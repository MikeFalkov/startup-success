# startup_success/dataset.py

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import typer

from startup_success.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

RAW_FILE = RAW_DATA_DIR / "big_startup_secsees_dataset.csv"
PROCESSED_FILE = PROCESSED_DATA_DIR / "dataset.csv"


@app.command()
def main(input_path: Path = RAW_FILE, output_path: Path = PROCESSED_FILE):
    logger.info(f"Reading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)

    logger.info("Cleaning and transforming dataset...")

    # Target
    df["success"] = df["status"].apply(lambda x: 0 if x == "closed" else 1)

    # Drop columns
    df.drop(columns=["permalink", "homepage_url", "name", "status", "state_code"], inplace=True)

    # Fill nulls
    df.fillna(
        {
            "category_list": "Unknown",
            "country_code": "Unknown",
            "region": "Unknown",
            "city": "Unknown",
        },
        inplace=True,
    )
    df.dropna(subset=["first_funding_at"], inplace=True)

    # Funding cleaning
    df["funding_total_usd"] = df["funding_total_usd"].replace("-", np.nan)
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")

    # Dates
    for col in ["founded_at", "first_funding_at", "last_funding_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["founded_year"] = df["founded_at"].dt.year
    df["first_funding_year"] = df["first_funding_at"].dt.year
    df["last_funding_year"] = df["last_funding_at"].dt.year
    df["days_to_first_funding"] = (df["first_funding_at"] - df["founded_at"]).dt.days
    df["funding_duration"] = (df["last_funding_at"] - df["first_funding_at"]).dt.days

    # Filter outliers
    df = df[df["funding_total_usd"] <= 5_000_000_000]
    df = df[(df["founded_year"] >= 1990) & (df["founded_year"] <= 2015)]
    df.dropna(subset=["days_to_first_funding", "funding_duration"], inplace=True)

    # Category clustering
    logger.info("Clustering category_list via TF-IDF + KMeans...")
    unique_categories = pd.Series(df["category_list"].dropna().unique())
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95, min_df=2)
    X_cat = vectorizer.fit_transform(unique_categories)

    kmeans = KMeans(n_clusters=20, random_state=42)
    clusters = kmeans.fit_predict(X_cat)
    cat_map = dict(zip(unique_categories, clusters))
    df["category_cluster"] = df["category_list"].map(cat_map)

    # Final drops
    df.drop(
        columns=["category_list", "founded_at", "first_funding_at", "last_funding_at"],
        inplace=True,
    )

    logger.info(f"Saving cleaned dataset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success("Dataset preprocessing complete.")


if __name__ == "__main__":
    app()
