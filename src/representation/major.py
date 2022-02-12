# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Sat 12 Feb 2022 02:22:02 AM CET
# Description: -
# ======================================================================
from typing import Dict, Callable

import numpy as np

from src.env.feature import Feature, Dataset
from src.util import lookup

class Major:
    def __init__(
        self,
        representation_dimension: int,
        representation_generator: Callable[[np.ndarray], np.ndarray],
        feature_values: Dict[Feature, np.ndarray],
        evaluation_function: Callable[[Dataset, Dict[Feature, np.ndarray]],float],
    ):

        self.feature_values = feature_values

        self.representation_dimension = representation_dimension
        self._representation_generator = representation_generator
        self._dataset_performance_lookup: Dict[Dataset, float] = dict()
        self._evaluation_function = evaluation_function

        self._feature_representation_lookup: Dict[Feature, np.ndarray] = dict()
        self._dataset_representations: Dict[Dataset, np.ndarray] = dict()

    def get_feature_representation(
        self, 
        feature: Feature,
        from_lookup: bool = False,
        lookup_ds_name: str = ""
    ) -> np.ndarray:
        if feature in self._feature_representation_lookup:
            return self._feature_representation_lookup[feature]

        else:
            if not from_lookup:
                values = self.feature_values[feature]
            else:
                _, values = lookup.get_ds(lookup_ds_name)
                values = values[feature]

            feature_representation = self._representation_generator(values)
            self._feature_representation_lookup[feature] = feature_representation
            return feature_representation

    def get_dataset_representation(
        self,
        ds: Dataset,
        from_lookup: bool = False,
        lookup_ds_name: str = ""
    ) -> np.ndarray:
        if ds in self._dataset_representations:
            return self._dataset_representations[ds]

        representations = [
            self.get_feature_representation(
                x, 
                from_lookup = from_lookup,
                lookup_ds_name = lookup_ds_name
            ) for x in ds.features
        ]
        representations = np.array(representations)
        self._dataset_representations[ds] = representations
        
        return representations

    def get_dataset_performance(self, ds: Dataset) -> float:
        if ds in self._dataset_performance_lookup:
            return self._dataset_performance_lookup[ds]
        else:
            performance = self._evaluation_function(ds, self.feature_values)
            self._dataset_performance_lookup[ds] = performance
            return performance
