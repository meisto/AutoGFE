# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 22 Dec 2021 02:56:53 PM CET
# Description: -
# ======================================================================
from typing import FrozenSet, Set

from src.env.feature import Transformation

def sanity_check(transformations: FrozenSet[Transformation]) -> bool:
    """
        Sanity check for features
    """

    # Name check
    names: Set[str] = set()
    for t in transformations:
        assert t.name not in names, "Multiple transformations share same name."
        names.add(t.name)

    for t in transformations:
        assert t.arity == len(t.required_datatypes), \
            f"Arity not consistent with required datatypes in feature '{t.name}'."

    return True
