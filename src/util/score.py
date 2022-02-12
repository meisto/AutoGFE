# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 22 Dec 2021 02:12:06 PM CET
# Description: -
# ======================================================================
from typing import Dict

import numpy as np
from sklearn import model_selection, linear_model, tree, ensemble

from src.env.feature import Dataset, Feature

def evaluate(
        dataset: Dataset,
        context: Dict[Feature, np.ndarray],
        modeltype: str,
    ) -> float:
    """
        Given data and labels, return the mean score of 
        a k-fold crossvalidation run.
    """
    assert modeltype in [
        "logistic_regression",
        "decision_tree",
        "random_forest",
    ]

    # If the dataset does not contain any features, return zero
    if len(dataset.features) <= 0:
        return 0.0

    data    = [context[x] for x in dataset.features]
    data    = np.array(data)
    labels  = context[dataset.labels]

    assert data.shape[0] == len(dataset.features)

    if modeltype == "logistic_regression":
        model = linear_model.LogisticRegression(max_iter=500)
    elif modeltype == "decision_tree":
        model = tree.DecisionTreeClassifier()
    elif modeltype == "random_forest":
        model = ensemble.RandomForestClassifier()
    else:
        raise Exception("This should not happen.")

    cross_val_score = model_selection.cross_val_score(
        estimator=model,
        X = data.T,
        y = labels,
        scoring = "f1_micro",
        cv = 8,
    )

    return cross_val_score.mean()
