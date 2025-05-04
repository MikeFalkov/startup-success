# Startup Success

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project is to predict whether a startup which is currently operating turn into a success or a failure.

---
## Authors

- **Michael Falkov**
- **Vladimir Skirmant**
---

## Model

The model is a `BalancedRandomForestClassifier` trained on SMOTE+Tomek-resampled data. Threshold optimization is applied using F1-score targeting the minority (failure) class.

- Features include funding history, founding timeline, and clustered category vectors.
- Predictions return both class labels and probabilities.

---

## Tools Used
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) – Standardized project structure
- [Typer](https://typer.tiangolo.com/) – CLI application framework
- [loguru](https://github.com/Delgan/loguru) – Structured logging
- [joblib](https://joblib.readthedocs.io/) – Model serialization
- 
- [scikit-learn](https://scikit-learn.org/) – ML models and preprocessing
- [imbalanced-learn](https://imbalanced-learn.org/) – SMOTE+Tomek for class imbalance


---

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         startup_success and configuration for tools like black.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── startup_success   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes startup_success a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models.
    │
    └── plots.py                <- Code to create visualizations
```

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-org/startup_success.git
cd startup_success
```

2. **Create a virtual environment**

```bash
make create_environment
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
make requirements
```

---

## Usage

### Clean & Process Data

```bash
make data
```

### Generate Features

```bash
make features
```

### Train Model

```bash
make train
```

### Predict

```bash
make predict
```

---

## Development

### Code Quality

Run lint and formatting checks with [ruff](https://docs.astral.sh/ruff/):

```bash
make lint    # Check formatting and lint
make format  # Auto-format and fix issues
```

### Cleanup

```bash
make clean          # Remove cache
make clean-data     # Remove interim and processed data
make clean-models   # Remove saved models
```
