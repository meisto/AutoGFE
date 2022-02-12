# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Sat 12 Feb 2022 11:07:19 AM CET
# Description: -
# ======================================================================
from typing import Dict, Set

from src.env.feature import Dataset, Transformation, Feature


import numpy as np

_dss: Dict[str, Dataset] = dict()
_dss_context: Dict[str, Dataset] = dict()

_transforms: Set[Transformation] = set()


def add_ds(ds: Dataset):
    dataset_name = ds.name

    assert ds not in _dss, "Dataset already registered."

    _dss[dataset_name] = ds

def add_ds_value(ds: Dataset, feature_values: Dict[Feature, np.ndarray]):
        if not (ds.name in _dss_context) and ds.name in _dss:
            _dss_context[ds.name] = feature_values

def get_ds(x: str):
    assert x in _dss, f"Unknown dataset '{x}'"
    assert x in _dss_context, f"No context."

    return _dss[x], _dss_context[x]

def add_transformations(t: Set[Transformation]):
    _transforms.update(t)

def get_all_transformations():
    return _transforms.copy()


