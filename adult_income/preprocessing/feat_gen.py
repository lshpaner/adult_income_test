################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

import os
import pandas as pd
import typer

from supportive_care.functions import mlflow_dumpArtifact, mlflow_loadArtifact

from supportive_care.constants import (
    target_death,
    var_index,
    epic_death,
    exp_artifact_name,
    preproc_run_name,
)

################################################################################
################ Model Preprocessing and Feature Engineering ###################
################################################################################

################################################################################
################ Step 2: Define Typer Application ##############################
################################################################################

app = typer.Typer()

################################################################################
################ Step 3: Define Main Function ##################################
################################################################################


@app.command()
def main(
    input_data_file: str = "./data/processed/df_sans_zero_missing.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Processes the input data file and generates feature space X and target variable y.

    Args:
        input_data_file (str): Path to the input parquet file.
    """

    ############################################################################
    ################ Step 4: Load Input Data ###################################
    ############################################################################

    # Read the input data file
    df = pd.read_parquet(input_data_file).set_index(var_index)
    print("-" * 80)
    print(f"# of DataFrame Columns: {df.shape[1]}")

    ############################################################################
    ################ Step 5: Training Stage ####################################
    ############################################################################

    if stage == "training":
        # Filter features: exclude columns with 'lead' and death-related terms,
        # except epic_death
        before = set(df.columns)
        X = df[[col for col in df.columns if "lead" not in col]].copy()
        X = X[
            [col for col in X.columns if "death" not in col.lower() and col != epic_death]
        ].copy()
        after = set(X.columns)
        print(f"# of `death` and `lead` vars from DataFrame: {len(before - after)}")
        print(f"Resulting # of Columns in X:  {X.shape[1]}")
        print("-" * 80)

        ############### Store Final List of Features for Production ##

        X_columns_list = X.columns.to_list()

    ############################################################################
    ################ Step 6: Inference Stage Load X_columns list ###############
    ############################################################################

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Features List From `feat_gen.py`
        ########################################################################
        # During training, we identified and stored `X_columns_list`.
        # Now, we reload this to ensure that inference follows the same
        # preprocessing pipeline as training, maintaining consistency.
        ########################################################################

        # Load feature column names from Mlflow
        X_columns_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,  # Use the same run_name as training
            obj_name="X_columns_list",
        )

        X = df[X_columns_list].copy()

        # dropping "_missing" cols strings will be created next
        X_columns_list = [col for col in X_columns_list if "_missing" not in col]

    ############################################################################
    ################ Step 7: Create and Append Missingness Indicators ##########
    ############################################################################

    # Create missing indicators for each feature
    X_missing = X.isna().astype(int)
    X_missing.columns = [f"{col}_missing" for col in X.columns]

    ############################################################################
    ################ Step 9: Store Final List of Features for Production #######
    ############################################################################
    if stage == "training":
        # Save feature column names to a pickle file
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,  # Consistent run_name for all artifacts
            obj_name="X_columns_list",
            obj=X_columns_list,
        )
        print(f"\nShape of X: {X.shape} \n")


    # Append missing indicators to feature space
    X = pd.concat([X, X_missing], axis=1)

    print(f"New X Shape (after adding missingness indicators): {X.shape}")
    if stage == "inference":
        print("\033[33mNumber of rows may vary due to Step 17 of `preprocessing.py`\033[0m")
    print("-" * 80)
    print(f"\nFeature Space\n{X.head()}\n")

    ############################################################################
    ################ Step 10: Generate Target Variable for Training #############
    ############################################################################

    if stage == "training":
        # Target variables from constants.py
        y = df[target_death]
        print(f"\nOutcomes (y): {y.columns.to_list()}")
        for target in target_death:
            y[[target]].to_parquet(
                os.path.join(data_path, f"y_{target}.parquet"),
            )
    ############################################################################
    ################ Step 11: Save Processed Feature Space #####################
    ############################################################################

    # Save the feature space (X) and target variables (y) to parquet files
    X.to_parquet(os.path.join(data_path, "X.parquet"))


################################################################################

if __name__ == "__main__":
    app()
