#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = startup_success
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check


## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset (clean and cluster categories)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) startup_success/dataset.py

## Generate features
.PHONY: features
features:
	$(PYTHON_INTERPRETER) startup_success/features.py

## Train model
.PHONY: train
train:
	$(PYTHON_INTERPRETER) startup_success/modeling/train.py

## Predict using trained model
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) startup_success/modeling/predict.py

## Remove saved models
.PHONY: clean-models
clean-models:
	rm -rf models/*

## Remove processed/interim data
.PHONY: clean-data
clean-data:
	rm -rf data/processed/* data/interim/*

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

## Show help message
help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
