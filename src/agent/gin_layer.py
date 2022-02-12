# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 26 Jan 2022 06:39:31 PM CET
# Description: -
# ======================================================================
from dataclasses import dataclass
from typing import Any, Dict, List, Set

import torch

from src.env.ttree import TTNode
from src.env.feature import Dataset

class GINLayer(torch.nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, epsilon: float, dimensionality: int):
        super().__init__()

        self.epsilon = epsilon
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dimensionality, dimensionality),
            torch.nn.ReLU()
        )

    def forward(
        self, 
        child_datasets: Dict[Dataset, Set[Dataset]],
        dataset_representation_lookup: Dict[Dataset, torch.Tensor]
    ) -> Dict[Dataset, torch.Tensor]:

       
        # Generate new representations for all nodes
        all_node_aggregations = []
        node_indices: Dict[TTNode, int] = {}
        for i, ds in enumerate(child_datasets.keys()):
            
            # Generate child values
            children = child_datasets[ds]
            children = [dataset_representation_lookup[x] for x in children]

            own_value = ( 1 + self.epsilon) * dataset_representation_lookup[ds]

            children.append(own_value)

            aggregated = torch.stack(children, dim=0).sum(dim=0)
            all_node_aggregations.append(aggregated + own_value)
            
            # Update indices for later retrieval
            node_indices[ds] = i


        # Apply MLP
        all_node_aggregations = torch.stack(all_node_aggregations, dim=0)
        node_representations = self.mlp(all_node_aggregations)

        # Res
        res = dataset_representation_lookup.copy()
        for x in child_datasets.keys():
            index = node_indices[x]
            slce = node_representations[index,:]
            res[x] = slce

        return res
