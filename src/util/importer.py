# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 22 Dec 2021 04:20:51 PM CET
# Description: -
# ======================================================================
import random
from typing import Set, Dict, Tuple

import numpy as np
import pandas as pd

from src.env.feature import Feature, Dataset


def get_datatype(s: pd.Series) -> str:
    t = s.dtype

    if t in [np.float32, np.float64]:
        return "REAL"
    if t in [np.int32, np.int64]:
        return "INT"
    elif t in []:
        return "CATEGORICAL"
    elif t in []:
        return "DATE"
    else:
        raise NotImplementedError(f"Unknown datatype '{t}'")

def get_tags(s: pd.Series, datatype: str) -> Set[str]:

    # Currently implemented tags:
    # NONZERO, POSITIVE, NEGATIVE

    if datatype == "REAL":
        tags = set()
        if (s > 0.0).all(): tags.add("POSITIVE")
        if (s < 0.0).all(): tags.add("NEGATIVE")
        if (s == 0.0).any(): 
            tags.add("HASZERO") 
        else:
            tags.add("NONZERO")
        return tags
    elif datatype == "INT":
        tags = set()
        if (s > 0).all(): tags.add("POSITIVE")
        if (s < 0).all(): tags.add("NEGATIVE")
        if (s == 0).any(): 
            tags.add("HASZERO")
        else:
            tags.add("NONZERO")
        return tags
    elif datatype == "CATEGORICAL":
        return set()
    elif datatype == "DATE":
        return set()
    else:
        raise NotImplementedError(f"Unknown datatype "\
            "'{datatype'. Can't tag.")

def cleanup_dataframe(
    df: pd.DataFrame,
    labels: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    
    # Shorten dataset if it is too large
    if df.shape[0] > 10000:
        indices = sorted(random.sample(range(df.shape[0]), k=10000))
        df = df.iloc[indices]
        labels = labels[indices]


    # Filter out all classes that have less than 8 datapoints
    vc = labels.value_counts()
    large_enough = list(vc[vc >= 12].index)
    selector = labels.isin(large_enough)

    df = df[selector]
    labels = labels[selector]

    # 
    assert "labels" not in df, "Labels should not be in dataframe"
        

    obects = df.select_dtypes(include=["object"])
    assert obects.empty, f"{str(obects)}"

    categorical = df.select_dtypes(include=["category"])
    if not categorical.empty:
        for c in categorical.columns:

            df = pd.concat([
                df,
                pd.get_dummies(df[c], prefix=c, dtype="float")
            ], axis=1)
            df.drop(columns=c, inplace=True)


    


    assert type(labels) == pd.Series,\
        f"Labels vector must be pd.Series, is '{type(labels)}'."

    return (df, labels)


def generate_dataset(
    df: pd.DataFrame,
    labels: pd.Series,
    name: str
) -> Tuple[Dataset, Dict[Feature, np.ndarray]]:
    """
    Given a pandas DataFrame, generate a DatasetRepresentation that can be used
    by this implementation.

    Note: The given DataFrame should already have been cleaned (no NaN, no 
    outliers, etc.).

    Returns:
        A dataset with feature,
        A dictionary mapping features to feature values
    """

    df, labels = cleanup_dataframe(df, labels)


    new_df = df.copy()

    # Generate Features
    renaming: Dict[str, Feature] = dict()
    features: Set[Feature] = set()
    for column in df.columns:
        series: pd.Series = df[column]

        # Identify feature type 
        # TODO add feature type identification for REAL, INT, DATE
        datatype: str = get_datatype(series)

        # Tagging
        tags: Set[str] = get_tags(series, datatype)

        # Generate and set feature
        f = Feature(
            datatype = datatype,
            tags = frozenset(tags),
            name = column
        )
        features.add(f)
        renaming[column] = f

    new_df.rename(columns=renaming, inplace=True)

    # Generate label
    labels_datatype: str = "CATEGORICAL"
    labels_tags: Set[str] = get_tags(labels, labels_datatype)
    labels_feature = Feature(
        datatype = labels_datatype,
        tags = frozenset(labels_tags),
        name = "labels"
    )

    new_df[labels_feature] = labels

    # Generate a lookup-table for values
    d: Dict[Feature, np.ndarray] = {}
    for c in new_df.columns:
        d[c] = new_df[c].values

    return (Dataset(frozenset(features),labels_feature, name), d)

