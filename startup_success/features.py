# startup_success/features.py

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from startup_success.config import PROCESSED_DATA_DIR

app = typer.Typer()

INPUT_FILE = PROCESSED_DATA_DIR / "dataset.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "features.csv"


@app.command()
def main(input_path: Path = INPUT_FILE, output_path: Path = OUTPUT_FILE):
    logger.info(f"Loading input features from: {input_path}")
    df = pd.read_csv(input_path)

    # Drop high-cardinality or leakage columns
    df.drop(columns=["city"], inplace=True, errors="ignore")

    # Optionally split into X/y if needed elsewhere
    logger.info(f"Saving final features to: {output_path}")
    df.to_csv(output_path, index=False)

    logger.success("Feature engineering complete.")


if __name__ == "__main__":
    app()
