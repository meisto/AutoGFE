# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Sat 18 Dec 2021 04:51:05 PM CET
# Description: -
# ======================================================================
# stdlib imports
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, FrozenSet, Dict, Optional
from uuid import UUID, uuid4 as get_id
import itertools

# library imports
import numpy as np

###############################################################################
###############################################################################
"""
    Definition of some ID wrappers, they could be replaced with only their
    UUID object, but wrapping enables typechecking.
"""
@dataclass(frozen=True)
class FeatureID:
    id: UUID = field(default_factory=get_id) # get_id()

    def __repr__(self):
        """
        Custom __repr__ for better overview during development.
        """
        return "FID"

@dataclass(frozen=True)
class TransformationID:
    id: UUID = field(default_factory=get_id)

    def __repr__(self):
        """
        Custom __repr__ for better overview during development.
        """
        return "TID"



###############################################################################
###############################################################################
"""
    Definition of some ID wrappers, they could be replaced with only their
    UUID object, but wrapping enables typechecking.
"""
@dataclass(init=True, frozen=True)
class Feature:
    datatype: str
    tags: FrozenSet[str]
    # values: pd.Series
    name: Optional[str] = field(default=None)
    id: FeatureID = field(default_factory=FeatureID)

    def __repr__(self):
        """
        Custom __repr__ for better overview during development.
        """
        n_repr = self.name
        dt_repr = self.datatype
        tag_repr = "{" + ",".join(self.tags) + "}"
        return f"Feature(name='{n_repr}',datatype={dt_repr},tags={tag_repr})"


@dataclass(init=True, frozen=True)
class Dataset:
    """
    Implementers Note: Dataset has no explicit DatasetID value, since the 
    dataclass implementer automatically generates a hash function that enables
    comparison.
    """
    features: FrozenSet[Feature]
    labels: Feature
    name: Optional[str] = field(default=None)

    def __post_init__(self):
        assert type(self.features) == frozenset, f"{type(self.features)} != {frozenset}"

    def __repr__(self):
        """
        Custom __repr__ for better overview during development.
        """

        names = [(x.name if x.name != None else "noname") for x in self.features]

        f_s = ",".join(names)
        return f"Dataset({f_s})"


    def __eq__(self, other):
        if isinstance(other, self.__class__):

            if self.labels != other.labels: return False
            if len(self.features) != len(other.features): return False

            for f in self.features: 
                if f not in other.features: return False
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply_transformation(
        self,
        features: List[Feature],
        transform, # :Transformation,
    ): # -> Dataset:

        new_feature     = transform.apply(features)
        new_features    = self.features.union({new_feature})

        return Dataset(new_features, self.labels, self.name)

    

@dataclass(init=True, frozen=True)
class Transformation:

    required_datatypes: Tuple[str,...]
    produced_datatype: str
    required_tags: FrozenSet[str]
    produced_tags: FrozenSet[str]
    arity: int
    name: str

    def __post_init__(self):
        """
            Function that is automatically called at the end of init.
            In this case used for consistency checks.
        """
        assert self.arity == len(self.required_datatypes),\
            "Arity incompatible with number of required datatypes in feature "\
            f"'{self.name}'."


    def apply(
        self,
        feats: Tuple[Feature, ...],
        context: Dict[Feature, np.ndarray],
    ) -> Feature:

        # Safety checks
        assert len(feats) == self.arity,\
            f"Can't apply {len(feats)} features to {self.arity}-ary transform."
        assert all(self.required_tags.issubset(f.tags) for f in feats),\
            "One or more features are invalid for this transformation"

        # Apply transformations
        f = [context[f] for f in feats]
        t = transformation_lambda_lookup[self]
        values: np.ndarray = t(f)

        # TODO: This type of produced tags does not really check out
        name_map = lambda x: str(x.name) if x.name != None else "noname"
        
        new_name = name_map(self) + "(" + ",".join(map(name_map, feats)) + ")"
        new_feature = Feature(
            self.produced_datatype,
            self.produced_tags,
            name=new_name
        )
        context[new_feature] = values

        return new_feature
    
    def apply_wo_context(
        self,
        feats: Tuple[Feature,...],
        context: Dict[Feature, np.ndarray],
    ) -> Tuple[Feature, np.ndarray]:
        # Safety checks
        assert len(feats) == self.arity,\
            f"Can't apply {len(feats)} features to {self.arity}-ary transform."
        assert all(self.required_tags.issubset(f.tags) for f in feats),\
            "One or more features are invalid for this transformation"

        # Apply transformations
        f = [context[f] for f in feats]
        t = transformation_lambda_lookup[self]
        values: np.ndarray = t(f)

        # TODO: This type of produced tags does not really check out
        name_map = lambda x: str(x.name) if x.name != None else "noname"
        
        new_name = name_map(self) + "(" + ",".join(map(name_map, feats)) + ")"
        new_feature = Feature(
            self.produced_datatype,
            self.produced_tags,
            name=new_name
        )

        return new_feature, values



    def get_possible_inputs(self, ds: Dataset) -> List[Tuple[Feature, ...]]:
        """
            Given a dataset return a set containing all possible inputs to this
            transformation.
        """

        possibilites = []
        for required_datatype in self.required_datatypes:
            fitting = {
                x for x in ds.features if 
                x.datatype == required_datatype and
                self.required_tags.issubset(x.tags)
            }
            possibilites.append(fitting)

        # Return cartesian product
        return list(itertools.product(*possibilites))

###############################################################################
###############################################################################
transformation_lambda_lookup: Dict[Transformation, Callable] = dict()
def register_transformation_lambda(tid: Transformation, val: Callable) -> None:
    assert not tid in transformation_lambda_lookup, \
        f"Value for transformation id '{tid}' already registered."
    transformation_lambda_lookup[tid] = val
