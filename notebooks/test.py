# %%
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd
import os
import sys

import equiboots as eqb

adult_x = pd.read_parquet("../data/processed/X.parquet")
adult_y = pd.read_parquet("../data/processed/y_income.parquet")

from adult_income.functions import find_best_model, mlflow_load_model


def return_best_model(outcome, metric, mlruns_location):

    outcome = "income"
    experiment_name = outcome + "_model"

    run_name, estimator_name = find_best_model(
        experiment_name, metric, mlruns_location=mlruns_location
    )

    model_name = f"{estimator_name}_{outcome}"
    best_model = mlflow_load_model(
        experiment_name, run_name, model_name, mlruns_location=mlruns_location
    )
    return best_model


best_model = return_best_model("income", "valid Average Precision", "../mlruns/models/")


test_x, test_y = best_model.get_test_data(adult_x, adult_y)

y_pred = best_model.predict(test_x)
y_prob = best_model.predict_proba(test_x)[:, 1]
y_true = test_y
fairness_df = test_x[["race", "sex"]]

fairness_df

eq = eqb.EquiBoots(
    y_true=y_true,
    y_prob=y_prob,
    y_pred=y_pred,
    fairness_df=fairness_df,
    fairness_vars=["race", "sex"],
)
eq.grouper(groupings_vars=["race", "sex"])
sliced_data = eq.slicer("race")
