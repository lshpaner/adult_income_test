################################################################################
# STEP 1: Import required libraries and modules
################################################################################
from pathlib import Path
import typer
import pandas as pd
import json

from adult_income.functions import mlflow_loadArtifact, mlflow_load_model
from adult_income.modeling.predict import find_best_model
from adult_income.constants import (
    shap_artifact_name,
    shap_run_name,
    shap_artifacts_data,
)

from adult_income.config import (
    PROCESSED_DATA_DIR,
)

from tqdm import tqdm

tqdm.pandas()

app = typer.Typer()


@app.command()
def main(
    outcome: str = "default_outcome",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y_income.parquet",
    metric_name: str = "valid AUC ROC",  # Metric to select the best model
    mode: str = "max",  # max for metrics where higher is better, min otherwise
    explanations_path: Path = "",
    shap_val_flag: int = 1,  # flag for whether or not to print vals next to feats.
    top_n: int = 5,  # top n feats.
    hold_out: str = "valid",  # holdout set; `valid` for validation
    # -----------------------------------------
):

    ################################################################################
    # STEP 2: Set up experiment parameters
    ################################################################################
    experiment_name = f"{outcome}_model"

    ################################################################################
    # STEP 3: Find and load the best model
    ################################################################################

    run_name, estimator_name = find_best_model(
        experiment_name,
        metric_name,
        mode,
    )

    model_name = f"{estimator_name}_{outcome}"  # retrieve best model_name

    # Load best model and assign it to variable called model
    model = mlflow_load_model(experiment_name, run_name, model_name)

    ################################################################################
    # STEP 4: Load Processed Data (Features & Labels)
    ################################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)
    y = y.squeeze()  # Ensure labels are in Series format

    ################################################################################
    # STEP 5: Split Process into Validation, and Test Sets
    # This gives the end-user the flexibility to run this script on validation
    # or test data
    ################################################################################

    if hold_out == "valid":
        X_holdout, y_holdout = model.get_valid_data(X, y)
    elif hold_out == "test":
        X_holdout, y_holdout = model.get_test_data(X, y)
    else:
        ValueError("Should be either valid or test")

    ################################################################################
    # STEP 6: Prepare the pipeline and transform data
    ################################################################################

    # Retrieve pipeline steps using built-in model_tuner getter
    X_holdout_transformed = (
        model.get_preprocessing_and_feature_selection_pipeline().transform(X_holdout)
    )

    ################################################################################
    # STEP 7: Load SHAP explainer
    ################################################################################
    # Load the SHAP explainer from artifact saved in explainer.py
    # using mlflow_dumpArtifact
    explainer = mlflow_loadArtifact(
        experiment_name=shap_artifact_name,
        run_name=shap_run_name,
        obj_name="explainer",
        artifacts_data_path=shap_artifacts_data,
    )

    ################################################################################
    # STEP 8: Compute SHAP values w/ progress bar
    ################################################################################
    print("Computing SHAP values...")
    with tqdm(total=X_holdout.shape[0], desc="SHAP Explaining") as pbar:
        shap_values = explainer(X_holdout_transformed)
        pbar.update(X_holdout.shape[0])

    ################################################################################
    # STEP 9: Process SHAP results
    ################################################################################

    # Extract transformed feature names from the preprocessing pipeline
    shap_feature_names = model.estimator.estimator[:-1].get_feature_names_out()

    # Convert SHAP values to DataFrame using transformed feature names
    shap_results = pd.DataFrame(
        shap_values.values,
        columns=shap_feature_names,
        index=X_holdout.index,
    )

    ################################################################################
    # STEP 10: Extract top n SHAP features (can be top any # based on make command)
    ################################################################################

    print(f"Extracting Top {top_n} SHAP features per patient...")

    # Get the top n features and their original SHAP values
    top_shap_pairs = shap_results.progress_apply(
        lambda row: row.abs().round(2).nlargest(top_n).to_json(),
        axis=1,
    )

    ################################################################################
    # STEP 11: Create SHAP DataFrame
    ################################################################################

    # Initialize a DataFrame to store SHAP output per patient
    shap_df = pd.DataFrame(index=X_holdout.index)

    if shap_val_flag:
        # Store top N features with their SHAP values as JSON strings
        joined = []
        for i in tqdm(range(top_shap_pairs.shape[0])):
            shap_feats = top_shap_pairs.iloc[i]
            joined.append(json.dumps(shap_feats))
        shap_df[f"Top {top_n} Features"] = joined

    else:
        # Store only the names of the top N features (without SHAP values)
        # In the if, feats are being added; here are just feat_names
        shap_df[f"Top {top_n} Features"] = top_shap_pairs.progress_apply(
            lambda x: ", ".join(map(str, list(json.loads(x).keys()))),
        )

    ################################################################################
    # STEP 12: Add confusion matrix and predictions metrics to dataframe
    ################################################################################
    # Generate predictions and predicted probabilities on the holdout set
    y_pred = model.predict(X_holdout, optimal_threshold=True)
    y_pred_proba = model.predict_proba(X_holdout)[:, 1]

    # Add confusion matrix components for each patient
    shap_df["TP"] = ((y_holdout == 1) & (y_pred == 1)).astype(int)
    shap_df["FN"] = ((y_holdout == 1) & (y_pred == 0)).astype(int)
    shap_df["FP"] = ((y_holdout == 0) & (y_pred == 1)).astype(int)
    shap_df["TN"] = ((y_holdout == 0) & (y_pred == 0)).astype(int)

    # Store true labels, predicted labels, predicted probabilities, and outcome name
    shap_df["y_true"] = y_holdout
    shap_df["y_pred"] = y_pred
    shap_df["y_pred_proba"] = y_pred_proba
    shap_df["Outcome"] = outcome

    ################################################################################
    # STEP 13: Save results to CSV file
    ################################################################################

    shap_df.to_csv(explanations_path, index=True)
    print(f"Results saved to '{explanations_path}'")


if __name__ == "__main__":
    app()
