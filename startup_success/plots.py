# startup_success/plots.py

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from startup_success.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

sns.set_theme(style="whitegrid")


def plot_founding_years(df: pd.DataFrame, output_path: Path):
    fig_path = output_path / "founding_years_hist.png"
    plt.figure(figsize=(12, 6))
    sns.histplot(df["founded_year"].dropna(), bins=50, kde=False, color="skyblue")
    plt.title("Distribution of Startup Founding Years")
    plt.xlabel("Year")
    plt.ylabel("Number of Startups")
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    logger.success(f"Saved founding years histogram: {fig_path}")


def plot_funding_box(df: pd.DataFrame, output_path: Path):
    fig_path = output_path / "funding_boxplot.png"
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df["funding_total_usd"].dropna())
    plt.title("Boxplot of Funding Total (USD)")
    plt.xlabel("Funding Total (USD)")
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    logger.success(f"Saved funding boxplot: {fig_path}")


def plot_category_clusters(df: pd.DataFrame, output_path: Path):
    if "category_cluster" not in df.columns:
        logger.warning("Missing 'category_cluster' column. Skipping category plot.")
        return
    fig_path = output_path / "category_cluster_counts.png"
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df, x="category_cluster", order=sorted(df["category_cluster"].dropna().unique())
    )
    plt.title("Distribution of Clustered Categories")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    logger.success(f"Saved category cluster distribution: {fig_path}")


@app.command()
def main(input_path: Path = PROCESSED_DATA_DIR / "dataset.csv", output_path: Path = FIGURES_DIR):
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    logger.info("Generating plots...")
    plot_founding_years(df, output_path)
    plot_funding_box(df, output_path)
    plot_category_clusters(df, output_path)
    logger.success("All plots generated.")


if __name__ == "__main__":
    app()
