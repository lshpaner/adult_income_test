Cookie Model Documentation (README)
=========================================

A cookie-cutter template integrating **EDA**, **model development**, and **fairness auditing**. This project enables fast and reproducible experimentation using the `model-tuner`, `eda-toolkit`, and `equiboots` Python libraries.

---

## Project Objective

This repository provides a reusable, cookie-cutter-style template for developing machine learning workflows that integrate **exploratory data analysis (EDA), model development, and fairness auditing**. It is designed to streamline experimentation and support reproducible pipelines using the **`model-tuner`**, **`eda-toolkit`**, and **`equiboots`** Python libraries.

> A sample binary classification task (UCI Adult Income Dataset) is included to demonstrate the template’s end-to-end capabilities.

---

## Dataset Summary

***Note:*** This is an example dataset included in the project for posterity and reproducibility. It provides a standardized baseline to illustrate the toolkit’s full workflow, from preprocessing to fairness auditing.

- **Dataset**: UCI Adult Income Dataset
- **Source**: UCI Machine Learning Repository  
- **Goal**: Predict income >$50K  
- **Sample Size**: ~48,000 adults (1994 US Census)  
- **Features**: Age, education, occupation, race, gender, hours worked, etc.  
- **Task Type**: Binary classification

---

## Key Features

### Initial Setup

- Predefined directory structure for easy replication
- Standardized preprocessing steps:
  - Remove zero-variance features
  - Handle missing values
  - Create missingness indicators
  - Feature engineering

### Modeling (via `model-tuner`)

- Supports `logistic regression`, `random forest`, `xgboost`, `catboost`
- Built-in:
  - Custom hyperparameter tuning (grid/random search)
  - Train/test/val split or k-fold
  - Early stopping
  - Model calibration
  - MLflow logging

### Evaluation

- All metrics automatically logged to **MLflow**
- Easy threshold selection based on precision-recall targets
- SHAP value analysis for model explainability

### Fairness Analysis (via `equiboots`)

- Disparity analysis across sensitive features (e.g., race, sex)
- Bootstrapped comparisons and pointwise metrics
- Visualizations: Violin plots, ROC disparities, etc.

---

## Installation

```bash
pip install -r requirements.txt
```

### ⚠️ Important Setup Instructions
To use this cookie-cutter template with editable modules:

  Ensure you have a `setup.py` in the root directory of the project. Example:

  ```python
  # setup.py
  from setuptools import find_packages, setup
  
  setup(
      name='src',
      packages=find_packages(),
      version='0.1.0',
      description='cookie-cutter data science re-adapted to be used with the `model-tuner`, `eda-toolkit`, and `equiboots` Python libraries.',
      author='Leonid Shpaner, Arthur Funnell, Panayiotis Petousis, UCLA CTSI',
      license='MIT',
  )
  ```

Project Organization
-------------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
