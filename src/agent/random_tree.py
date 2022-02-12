# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Thu 10 Feb 2022 12:07:11 AM CET
# Description: -
# ======================================================================
# stdlib imports
from typing import FrozenSet, Dict
import random

# library imports
import numpy as np

# Local file imports
from src.env.feature import Dataset, Transformation, Feature
from src.env.ttree import TTree

def create_random_tree(
    root_dataset: Dataset,
    transformations: FrozenSet[Transformation],
    number_expansions: int,
    context: Dict[Feature, np.ndarray],
) -> TTree:
    """
        Return a new transformation tree rooted at dataset root_dataset.
    """
    tree = TTree(root_dataset)
    
    # Change state of tree
    explore_subtree(
        root_dataset,
        transformations,
        number_expansions,
        tree,
        context,
    )

    return tree

def explore_subtree(
    root_dataset: Dataset,
    transformations: FrozenSet[Transformation],
    number_expansions: int,
    tree: TTree,
    context: Dict[Feature, np.ndarray],
):
    """
        Further explore a subtree rooted at node root_dataset.
    """

    # 
    tree = TTree(root_dataset)
    produced_datasets = {root_dataset}

    # Probs of expansion vs reduction
    prob_transform = len(transformations) / (len(transformations) + 1)
    # prob_reduction = 1 - prob_transform

    for _ in range(number_expansions):
        
        if random.random() <= prob_transform:
            # Transformation

            # Sample a random dataset
            ds = random.sample(produced_datasets, 1)[0]
            
            # Sample a random transformation
            transform = random.sample(transformations, 1)[0]

            # Sample a random set of input features
            possible_inputs = transform.get_possible_inputs(ds)
            if len(possible_inputs) == 0: continue
            features = random.choice(possible_inputs)

            new_ds = tree.transform_dataset(
                parent_dataset  = ds,
                transformation  = transform,
                feature_sets    = frozenset([features]),
                context         = context
            )
            produced_datasets.add(new_ds)
        else:
            # Reduction

            # Sample a random dataset
            ds = random.sample(produced_datasets, 1)[0]

            # Randomly generate the number of reduced features
            number_samples = int(random.random() * len(ds.features))

            # Sample a random set of input features
            features = random.sample(ds.features, k=number_samples)
            features = frozenset(features)

            new_ds = tree.reduce_dataset(ds, features)
            produced_datasets.add(new_ds)
