# ML-Decision_Tree

A Python implementation of decision trees for classification, with ready-to-use training/validation datasets and a requirements file for reproducibility. The repository includes a single main script, multiple TSV datasets (small, education, heart), example output folder, and a course handout PDF.

Languages: Python (100%)

## Table of Contents
- Overview
- Repository layout
- Setup
- Usage
- Datasets
- Example workflow
- Development notes
- License

## Overview

This project focuses on implementing and evaluating decision tree classifiers on several toy and real-world datasets (e.g., education outcomes, heart disease). The main entry point is `decision_tree.py`, which reads TSV-formatted datasets and outputs model performance and/or artifacts.

## Repository layout

Top-level:
- [.gitattributes](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/.gitattributes) — Git attributes.
- [F25_CS522_HW1.pdf](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/F25_CS522_HW1.pdf) — Course handout describing the assignment and requirements.

Programming directory: [Programming/](https://github.com/dhrumilp12/ML-Decision_Tree/tree/main/Programming)
- [decision_tree.py](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/Programming/decision_tree.py) — Main script implementing the decision tree(s).
- [requirements.txt](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/Programming/requirements.txt) — Python dependencies for the project.
- Datasets (TSV):
  - [small_train.tsv](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/Programming/small_train.tsv)
  - [small_val.tsv](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/Programming/small_val.tsv)
  - [education_train.tsv](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/Programming/education_train.tsv)
  - [education_val.tsv](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/Programming/education_val.tsv)
  - [heart_train.tsv](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/Programming/heart_train.tsv)
  - [heart_val.tsv](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/Programming/heart_val.tsv)
- Example outputs:
  - [example_output/](https://github.com/dhrumilp12/ML-Decision_Tree/tree/main/Programming/example_output) — Placeholder directory for sample results.
- macOS metadata:
  - [Programming/.DS_Store](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/Programming/.DS_Store) — Can be ignored.

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/dhrumilp12/ML-Decision_Tree.git
   cd ML-Decision_Tree/Programming
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```sh
   python -m venv .venv
   # Linux/macOS:
   . .venv/bin/activate
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

Note: If `requirements.txt` references specific versions, use them for consistent behavior.

## Usage

The typical workflow is to run `decision_tree.py` with a chosen train and validation TSV dataset. If the script supports command-line arguments (e.g., paths, hyperparameters), adjust accordingly. If not documented, you can start by editing/inspecting `decision_tree.py` and adapting the dataset paths inside the script.

Examples (adjust to match the script’s interface):

- Using the small dataset:
  ```sh
  python decision_tree.py --train small_train.tsv --val small_val.tsv
  ```

- Using the education dataset:
  ```sh
  python decision_tree.py --train education_train.tsv --val education_val.tsv
  ```

- Using the heart dataset:
  ```sh
  python decision_tree.py --train heart_train.tsv --val heart_val.tsv
  ```

If the script reads default paths (without CLI flags), place your terminal in the `Programming/` directory and run:
```sh
python decision_tree.py
```

Outputs may be printed to the console or written into `example_output/`. Check the script for exact behavior.

## Datasets

All datasets are TSV (tab-separated values) files split into train and validation sets:
- Small: minimal dataset for sanity checks (`small_train.tsv`, `small_val.tsv`)
- Education: categorical/numeric features for an education-related task (`education_train.tsv`, `education_val.tsv`)
- Heart: heart disease-related features, suitable for binary classification (`heart_train.tsv`, `heart_val.tsv`)

Inspect the TSV files to understand column names, targets, and feature types. Ensure that the script expects headers or a specific schema.

## Example workflow

1. Train on the education dataset and evaluate on validation:
   ```sh
   cd Programming
   python decision_tree.py --train education_train.tsv --val education_val.tsv
   ```
2. Review printed metrics (e.g., accuracy) or generated artifacts (trees, logs) under `example_output/` if produced.
3. Repeat for heart and small datasets to compare behavior.

## Development notes

- `decision_tree.py` is the main extension point:
  - Add CLI (`argparse`) for specifying train/val paths, depth limits, impurity criteria (Gini/entropy), pruning options, and random seeds.
  - Implement evaluation metrics (accuracy, precision/recall, confusion matrix).
  - Support categorical handling (one-hot or native splits) and missing values if needed.
- Reproducibility:
  - Pin versions in `requirements.txt`.
  - Set a random seed for deterministic splits or model behavior.
- Outputs:
  - Save results into `example_output/` (e.g., metrics JSON, plotted trees).
  - Consider logging to CSV or Markdown for assignment submission.

## License

No explicit license file is present. If you intend to share or modify this project, consider adding a LICENSE file (e.g., MIT, Apache-2.0, GPL-3.0) to clarify usage terms.

---
For assignment details and expected deliverables, see [F25_CS522_HW1.pdf](https://github.com/dhrumilp12/ML-Decision_Tree/blob/main/F25_CS522_HW1.pdf).
