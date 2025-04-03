################################################################################
######################### Import Requisite Libraries ###########################
import os
import typer
import pandas as pd

# import pickling scripts
from model_tuner.pickleObjects import dumpObjects

################################################################################

from adult_income.constants import (
    var_index,
    preproc_run_name,
    exp_artifact_name,
    miss_col_thresh,
    miss_row_thresh,
    percent_miss,
)

# import all user-defined functions and constants
from adult_income.functions import (
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    safe_to_numeric,
)

app = typer.Typer()


@app.command()
def main(
    input_data_file: str = "./data/raw/df.parquet",
    output_data_file: str = "./data/processed/df_sans_zero_missing.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Main script execution replacing sys.argv with typer.

    Args:
        input_data_file (str): Path to the input parquet file.
        output_data_file (str): Path to save the processed parquet file.
        stage (str): Processing stage (e.g., 'training' or 'inference').
    """
    ############################################################################
    # Step 1. Read the input data file
    ############################################################################

    df = pd.read_parquet(input_data_file)

    try:
        df.set_index(var_index)
    except:
        print("Index already set or 'var_index' doesn't exist in dataframe")

    if stage == "training":

        df_object = df.select_dtypes("object")
        print()
        print(
            "The following columns have strings and should be removed from "
            "modeling: \n \n"
            f"{df_object.columns.to_list()}. \n There are {df_object.shape[1]} of them."
        )

        ########################################################################
        # Step 2. String Columns Handling
        ########################################################################
        # String columns are identified and should be removed before modeling
        # because machine learning models typically require numerical inputs.
        # Keeping string columns in the dataset may lead to errors or
        # unintended behavior unless explicitly encoded.
        #
        # To ensure consistency between training and inference,
        # we save the list of string columns and track it using MLflow.
        ########################################################################

        # Extract column names to a list
        string_cols_list = df_object.columns.to_list()

        ########################################################################
        # Step 3. Save and Log String Column List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        # This list of string columns is dumped (stored) only to inform of what
        # the string columns are; no further action is taken; we do not need to
        # load this list into production, since it is only there for us to
        # see what the columns are.
        ########################################################################

        # Dump the string_cols_list into a pickle file for future reference
        dumpObjects(
            string_cols_list,
            os.path.join(data_path, "string_cols_list.pkl"),
        )

        # Log the string column list as an artifact in MLflow
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="string_cols_list",
            obj=string_cols_list,
        )

    ############################################################################
    ###################### Re-engineering Selected Features ####################
    ############################################################################

    ########################################################################
    # Step 4. Ensure Numeric Data and Feature Engineering
    ########################################################################
    # Convert any possible numeric values that may have been incorrectly
    # classified as non-numeric. This avoids accidental labeling errors.
    # Perform necessary feature transformations (if and as applicable), such as:
    # - Deriving weight in pounds from kilograms
    # - Calculating height in feet using BMI and weight
    # - Dropping redundant features to prevent overfitting
    ########################################################################

    # Convert possible numeric columns to actual numeric types
    df = df.apply(lambda x: safe_to_numeric(x))

    ############################################################################
    # Step 5. Process Additional Feature(s) (e.g., marital status)
    ############################################################################
    # Identify the top 10 most frequent marital status categories and retain only
    # these. Less common categories are replaced with a generic 'other' label to
    # maintain a controlled feature space and avoid overfitting due to rare
    # categories.
    ############################################################################

    if stage == "training":
        # Extract marital status feature
        marital_status = df["marital-status"]

        ############################################################################
        # Step 6. Log Top 10 Marital Status Categories and Replace Rare Ones
        ############################################################################

        # Save the list of string columns for consistency across training and
        # inference. Log the top 10 most frequent marital status values using MLflow.
        # Replace all other less common values with 'other' to simplify the feature
        # space and improve generalizability.

        ############################################################################

        # Dump the object into a pickle file
        dumpObjects(
            marital_status,
            os.path.join(data_path, "marital_status.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="marital_status",
            obj=marital_status,
        )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Object
        ########################################################################
        # During training, we identified and stored marital_status.
        # Now, we reload this to ensure that inference follows the same
        # preprocessing pipeline as training, maintaining consistency.
        ########################################################################

        ## Load marital_status from artifacts
        marital_status = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="marital_status",
        )

    ################################################################################
    # Step 7. Zero Variance Columns
    ################################################################################

    # Select only numeric columns s/t .var() can be applied since you can only
    # call this function on numeric columns; otherwise, if you include a mix
    # (object and numeric), it will throw the following FutureWarning:
    # Dropping of nuisance columns in DataFrame reductions
    # (with 'numeric_only=None') is deprecated; in a future version this will
    # raise TypeError.  Select only valid columns before calling the reduction.

    ################################################################################

    if stage == "training":
        # Extract numeric columns to compute variance and identify
        # zero-variance features
        numeric_cols = df.select_dtypes(include=["number"]).columns
        var_indf = df[numeric_cols].var()

        # identify zero variance columns
        zero_var = var_indf[var_indf == 0]
        # capture zero-variance cols in list
        zero_varlist_list = list(zero_var.index)

        ########################################################################
        # Step 8. Save and Log Zero Variance Columns List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        ########################################################################

        dumpObjects(
            zero_varlist_list,
            os.path.join(data_path, "zero_varlist_list.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
            obj=zero_varlist_list,
        )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Zero Variance Columns List
        ########################################################################

        # load zero_var_list
        zero_varlist_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
        )

    ########################################################################
    # Step 9. Remove zero variance cols from main df, and assign to new var
    # df_sans_zero
    ########################################################################
    df_sans_zero = df.drop(columns=zero_varlist_list)

    print(f"Sans Zero Var Shape: {df_sans_zero.shape}")

    print()
    print(f"Original shape: {df.shape[1]} columns.")
    print(f"Reduced by {df.shape[1]-df_sans_zero.shape[1]} zero variance columns.")
    print(f"Now there are {df_sans_zero.shape[1]} columns.")
    print()

    ############################################################################
    # Step 10. Handle Missing Data
    ############################################################################

    # Calculate the percentage of missing values for each column in df_sans_zero
    # 1. df_sans_zero.isnull().sum() counts the number of missing values in each
    #    column.
    # 2. len(df_sans_zero) gives the total number of rows in the DataFrame.
    # 3. Dividing the number of missing values by the total number of rows gives
    #    the proportion of missing values.
    # 4. Multiplying by 100 converts this proportion into a percentage.

    if stage == "training":

        """
        Process Description: Handling Missing Data in a Dataset

        1. Identifying Missing Data by Column

        The first step in handling missing data involves calculating the
        percentage of missing values for each column in the dataset. This helps
        visualize the distribution of missing values and determine a reasonable
        threshold for column retention.
        """

        # Compute the percentage of missing values for each column
        perc_missing_vals_per_col = (
            df_sans_zero.isnull().sum() / len(df_sans_zero)
        ) * 100

        # Select columns where less than 60% of the data is missing.
        """
        Filtering Columns Based on Missing Data Threshold

        Columns with more than 60% missing values were flagged for potential 
        removal, while those with a lower percentage were retained for further 
        analysis.

        After visualizing the missing data distribution, columns where more than 
        60% of values are missing are removed.

        """
        perc_below_indiv = perc_missing_vals_per_col[
            perc_missing_vals_per_col <= miss_col_thresh
        ].index.tolist()

        # Dump the perc_below_indiv into a pickle file
        dumpObjects(
            perc_below_indiv,
            os.path.join(data_path, f"{perc_below_indiv}.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name=perc_below_indiv,
            obj=perc_below_indiv,
        )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Percentage Below Threshold List
        ########################################################################

        # load perc_below_indiv
        perc_below_indiv = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name=perc_below_indiv,
        )

    # Create a new DataFrame including only columns where less than 60% of the
    # data is missing.
    df_sans_zero_missing = df_sans_zero.loc[:, perc_below_indiv]

    print(f"Sans Zero Missing 60% Missing Data: {df_sans_zero_missing.shape}")

    ############################################################################
    # Step 11. Remove Missingness in Rows

    # Filtering Rows Based on Missing Data and Specific Conditions

    # Once high-missing-value columns are removed, additional filtering is
    # applied at the row level. The goal is to remove rows that:

    # 1. Have zero recorded values for a critical feature (e.g., martial-status).
    # 2. Have a high percentage of missing values across all columns.

    # Rows where martial-status == 0 and a large proportion of other data
    # is missing (exceeding a predefined threshold miss_row_thresh) are removed.

    # This step ensures that rows with insufficient data for analysis are
    # discarded while keeping informative rows.

    ############################################################################

    # This is done only once in production for training
    # Apply the filtering (removing rows based on conditions)

    if stage == "training":

        # Get the number of rows before applying the filter
        rows_before = df_sans_zero_missing.shape[0]

        df_sans_zero_missing = df_sans_zero_missing[
            ~(
                (df_sans_zero_missing["capital-gain"] == 0)
                & (
                    (
                        df_sans_zero_missing.isnull().sum(axis=1)
                        / df_sans_zero_missing.shape[1]
                    )
                    > miss_row_thresh
                )
            )
        ]

        # Get the number of rows after applying the filter
        rows_after = df_sans_zero_missing.shape[0]

        # Calculate the number of rows removed
        rows_removed = rows_before - rows_after

        print(f"Number of rows removed: {rows_removed}")

    ############################################################################
    # Step 12. Calculate Row-wise Missingness Percentage
    ############################################################################
    # This step computes the proportion of missing values for each row in the
    # DataFrame. It helps identify rows with a high level of incompleteness, which
    # may be useful for filtering, imputation strategies, or downstream analysis.
    #
    # A new column is added to `df_sans_zero_missing` where each value represents
    # the percentage of columns that are missing for that row.
    ############################################################################

    df_sans_zero_missing[percent_miss] = df_sans_zero_missing.isna().mean(axis=1)

    ############################################################################
    # Step 13. Save Processed Data
    ############################################################################

    # Save out the dataframe to parquet file
    print(df_sans_zero_missing.shape)
    df_sans_zero_missing.reset_index().to_parquet(output_data_file)


if __name__ == "__main__":
    app()
