# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Mon 13 Dec 2021 06:17:52 PM CET
# Description: -
# ======================================================================
# stdlib imports
from dataclasses import dataclass, field
from typing import Set, Tuple, Dict, FrozenSet

# library imports
import numpy as np

# local file imports
from .feature import Feature, Transformation, Dataset

###############################################################################
###############################################################################
"""
    Define the Transformation Tree and its nodes.
"""
class TTEdge:
    """
        (Not really) abstract edge class.
    """
    def __init__(self):
        pass

@dataclass(frozen=True)
class TTEdgeExpand(TTEdge):
    """
        An expansion-type edge in the transformation tree.
    """
    transformation: Transformation
    features: FrozenSet[Tuple[Feature,...]]

    def __repr__(self):
        # s = [f.name if f.name != None else "noname" for f in self.features]
        # s = ", ".join(s)
        return f"Edge(expanding, -)"

@dataclass(frozen=True)
class TTEdgeReduce(TTEdge):
    """
        An reduction-type edge in the transformation tree.
    """
    features: FrozenSet[Feature]

    def __repr__(self):

        fn = [x.name if x.name != None else "noname" for x in self.features]
        fn = ",".join(fn)
        return f"Edge(reducing, removed=[{fn}])"


@dataclass(frozen=True)
class TTNode:
    """
        A node in the transformation tree
    """

    dataset: Dataset
    children: Dict = field(default_factory=dict) # Dict[TTEdge, TTNode]

    def __repr__(self):
        ds_name = str(self.dataset)
        num_children = len(self.children)
        return f"TTNode(dataset={ds_name}, num_children={num_children})"

# @dataclass(frozen=True)
class TTree:
    def __init__(self, root_dataset: Dataset):
        """
            A tree structure representing an transformation tree.

            This structure is implemented in a way that it should limit the 
            number of unnecessary duplicate calculations.
        """
        # Add features from root_dataset to list of all features
        self.features: Set[Feature] = set(root_dataset.features)

        # Protocol all generated features to avoid unnecessary duplicates
        self.generated_features: \
            Dict[Tuple[Tuple[Feature,...], Transformation], Feature] = dict()

        # Add lookup for number
        self.number_datasets = 0
        self.dataset_name_lookup: Dict[Dataset, int] = dict()

        # Add node lookup
        self.node_lookup: Dict[Dataset, TTNode] = dict()

        # Add a root node
        self.root_node: TTNode = self.register_dataset(root_dataset)

        # Add null-node as catchall
        null_dataset = Dataset(frozenset({}), root_dataset.labels)
        self.null_node = self.register_dataset(null_dataset)

    def register_dataset(
        self,
        ds: Dataset,
    ) -> TTNode:
        """
            Register a dataset in the transformation tree. If the dataset 
            exists already in the tree, return its node. Otherwise return 
            a new node.
        """

        # Get or generate new node with the new dataset
        if ds in self.node_lookup:
            return self.node_lookup[ds]
        else:
            new_node: TTNode = TTNode(ds)

            self.features.update(ds.features)
            self.node_lookup[new_node.dataset] = new_node

            self.dataset_name_lookup[ds] = self.number_datasets
            self.number_datasets += 1

            return new_node

    def transform_dataset(
        self,
        parent_dataset: Dataset,
        transformation: Transformation,
        feature_sets: FrozenSet[Tuple[Feature,...]],
        context: Dict[Feature, np.ndarray],
    ) -> Dataset:
        """
            Transform a dataset to generate a new dataset/node in the tree.
            Takes as input the node-to-be-transformed, a transformations,
            a set of features and the context (a dict with a features as keys
            and their values as values).

            Returns the new dataset. New dataset is also added to the tree if
            it is not there yet.
        """

        # Check for consistency
        assert parent_dataset in self.node_lookup.keys(), "Unknown dataset."
        for features in feature_sets:
            for f in features:
                assert f in self.features, f"Unknown feature '{f}'"
        
        # Get parent node and create an edge
        parent_node = self.node_lookup[parent_dataset]

        edge = TTEdgeExpand(transformation, feature_sets)

        # Check if such an edge already exists for this node
        if edge in parent_node.children.keys():
            # Node has already been explored
            return parent_node.children[edge].dataset

        # Generate new dataset
        new_features = set()
        for features in feature_sets:

            # Get new feature if it has already been generated, else generate
            if (features, Transformation) in self.generated_features:
                # Feature has already been generated and can be looked up
                new_feature: Feature = \
                    self.generated_features[(features, transformation)]
            else:
                # Feature must be newly generated
                new_feature = transformation.apply(features, context)
                self.features.add(new_feature)
                self.generated_features[(features, transformation)] = new_feature

            new_features.add(new_feature)


        all_features = parent_dataset.features.union(new_features)
        new_dataset: Dataset = Dataset(
            all_features,
            parent_dataset.labels,
            parent_dataset.name
        )

        new_node = self.register_dataset(new_dataset)
        parent_node.children[edge] = new_node

        return new_node.dataset

    def reduce_dataset(
        self,
        parent_dataset: Dataset,
        features: FrozenSet[Feature],
    ) -> Dataset:
        """
            Remove features from a dataset to generate a new dataset/node 
            in the tree.
            Takes as input the node-to-be-reduced and a set of features.

            Returns the reduced dataset. New dataset is also added to the tree 
            if it is not there yet.
        """
        # Check for consistency
        assert parent_dataset in self.node_lookup.keys(), "Unknown dataset."
        assert all(f in self.features for f in features), "Unknown feature"

        parent_node = self.node_lookup[parent_dataset]
        assert features.issubset(parent_node.dataset.features)

        edge = TTEdgeReduce(features)

        # Such an Edge exists already for this node
        if edge in parent_node.children.keys():
            # Node has already been explored
            return parent_node.children[edge].dataset

        new_features = parent_dataset.features.difference(features)
        new_dataset: Dataset = \
            Dataset(new_features, parent_dataset.labels, parent_dataset.name)

        new_node = self.register_dataset(new_dataset)
        parent_node.children[edge] = new_node

        return new_node.dataset
