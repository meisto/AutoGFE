# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Thu 06 Jan 2022 08:21:58 PM CET
# Description: -
# ======================================================================
# stlib imports
from typing import List, Dict, Callable

# Downloaded imports
import numpy as np
import pandas as pd

# Local imports
from src.env import feature


#TODO: This should probably go somewhere else
# Define a short function, because it looks better than a long lambda
def random_sampling(
    feature_values: np.ndarray,
    representation_dimension: int,
) -> np.ndarray:
    """

    """
    assert type(feature_values) == np.ndarray, f"Wrong datatype '{type(feature_values)}'."
    fmin = feature_values.min()
    fmax = feature_values.max()

    if fmin != fmax:
        normalized = (feature_values - fmin) / (fmax - fmin)
    else:
        normalized = feature_values - fmin

    samples = np.random.choice(normalized, representation_dimension)
    samples.sort()

    return samples


def quantile_data_sketch(
    values: np.ndarray,
    representation_dimension: int,
) -> np.ndarray:
    """
    TODO: Do
    """
    assert type(values) == np.ndarray, "Wrong datatype."

    std = values.std()
    if std <= 1e-4:
        res: List[float] = [0] * representation_dimension
        res[0] = 1.0

        return np.array(res, dtype=float)


    else:

        min_element = values.min()
        max_element = values.max()

        series = (values - min_element) / (max_element - min_element)


        labels = [i / representation_dimension for i in range(representation_dimension)]
        series = pd.cut(
            series, 
            bins=representation_dimension,
            labels=labels,
        ).value_counts().sort_index()
        
        series = series / series.sum()

        series = series.values
        assert type(series) == np.ndarray
        assert not np.isnan(series).any()
        assert len(series.shape) == 1, f"{series.shape}"


        return series

def gaussian(
    feature_values: pd.Series,
) -> np.ndarray:

    fmin = feature_values.min()
    fmax = feature_values.max()

    if fmin != fmax:
        normalized = (feature_values - fmin) / (fmax - fmin)
    else:
        normalized = feature_values - fmin

    values = normalized.values
    return np.array([values.mean(), values.std()])
    

def representation_lookup(
    name: str,
    representation_dimension: int,
) -> Callable[[pd.Series], np.ndarray]:
    lookup = {
        "quantile_data_sketch"  : lambda x: quantile_data_sketch(x, representation_dimension),
        "quantile_sketch_array" : lambda x: quantile_data_sketch(x, representation_dimension),
        "random_sampling"       : lambda x: random_sampling(x, representation_dimension),
        "gaussian"              : lambda x: gaussian(x),
    }
    return lookup[name]
