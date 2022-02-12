## ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Mon 13 Dec 2021 05:20:40 PM CET
# Description: -
# ======================================================================
# Imports from stdlib
from typing import Callable, Tuple, Dict, Optional
import random

# Imports from installed libraries
import gym
import numpy    as np

# Imports from local files
from src.representation.major import Major
from .feature import Transformation, Dataset, Feature
from . import ttree
from .reward import get_reward_function

class TTEnv(gym.Env):
    """
        TOOD
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        root_dataset: Dataset,
        representation_dimension: int,
        representation_generator: Callable[[np.ndarray], np.ndarray],
        evaluation_function: Callable[[Dataset, Dict[Feature, np.ndarray]],float],
        max_steps: int,
        feature_values: Dict[Feature, np.ndarray],
        reward_function: str,
    ):
        """
        Params:
            
        """
        super(TTEnv, self).__init__()

        assert representation_dimension > 0, "TODO: Write message"


        self.max_steps = max_steps
   
        # Set hyperparameter
        self._reward_function       = get_reward_function(reward_function)

        # Major
        self.major = Major(
            representation_dimension = representation_dimension,
            representation_generator = representation_generator,
            feature_values = feature_values,
            evaluation_function   = evaluation_function
        )


        # Root dataset
        self.root_dataset: Dataset = root_dataset
        self.root_performance = self.major.get_dataset_performance(root_dataset)

        self.tree: ttree.TTree = ttree.TTree(self.root_dataset)
        self.current_dataset: Dataset = self.root_dataset
        
        # Call reset to ensure everything is set up
        self.reset()

    def reset(self) -> np.ndarray:
        """
        TODO
    
        Returns:
            initial observation - np.array
        """

        self.steps = 0

        self.best_performance = self.root_performance
        self.prev_performance = self.root_performance

        self.current_dataset = self.root_dataset

        return self.tree.node_lookup[self.current_dataset]


    def step(
        self, 
        transformation_or_reduction: Optional[Transformation],
    ) -> Tuple[ttree.TTNode, float, bool, Dict]:
        """
        TODO

        Argumnet:
            action: int

        Returns:
            tuple with observation, reward, done, info
        """
        k = 4

        if transformation_or_reduction != None:
            # An expanding edge
            trans: Transformation = transformation_or_reduction

            # Get all possible inputs
            possible_inputs = trans.get_possible_inputs(self.current_dataset)
            possible_inputs = list(possible_inputs)

            if len(possible_inputs) == 0:
                new_dataset = self.current_dataset
            else:

                if len(possible_inputs) <= k:
                    subset = possible_inputs
                else:
                    subset = random.sample(possible_inputs, k=k)
 
                # Generate and add all new features to the subset
                new_dataset = self.tree.transform_dataset(
                    parent_dataset  = self.current_dataset,
                    transformation  = trans,
                    feature_sets    = frozenset(subset),
                    context         = self.major.feature_values,
                )

        else:
            # An reduction edge

            current_features = self.current_dataset.features
            current_labels = self.current_dataset.labels

            worst_feature: Feature      = random.sample(current_features,1)[0]
            worst_performance: float    = 1.0

    
            for f in current_features:

                reduced = current_features.difference({f})

                reduced = Dataset(reduced, current_labels)
                performance_reduced = self.major.get_dataset_performance(reduced)

                if performance_reduced < worst_performance:
                    worst_feature       = f
                    worst_performance   = performance_reduced

            new_dataset = self.tree.reduce_dataset(
                parent_dataset  = self.current_dataset,
                features        = frozenset({worst_feature}),
            )

        self.steps += 1

        # Update current dataset
        self.current_dataset = new_dataset

        # Calculate performance
        performance = self.major.get_dataset_performance(self.current_dataset)

        # Check if done
        if self.steps >= self.max_steps or performance == 1 or \
            self.current_dataset == self.tree.null_node.dataset:
            done = True
        else:
            done = False


        reward = self._reward_function(
            current_performance	            = performance,
            previous_performance	        = self.prev_performance,
            best_performance	            = self.best_performance,
        )

        # Update measures of performance
        self.best_performance = max(self.best_performance, performance)
        self.prev_performance = performance
        
        # Return performance metric
        info = {"performance": performance}

        current_node = self.tree.node_lookup[self.current_dataset]

        return (current_node, reward, done, info)
