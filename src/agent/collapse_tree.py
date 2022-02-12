# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Thu 10 Feb 2022 08:27:02 PM CET
# Description: -
# ======================================================================
from typing import Dict, List, Callable, Set
from collections import defaultdict

import torch
import numpy as np


from src.env.feature import Feature, Dataset
from src.env.ttree import TTree, TTNode
from src.env.ttenv import TTEnv
from src.representation.major import Major
from .gin_layer import GINLayer

def get_relevant_datasets(root_node: TTNode, k:int) -> Dict[Dataset, Set]:
    # Generate a set of relevant nodes all in reach k

    d = defaultdict(lambda: set())

    current_nodes = [root_node]
    d[root_node.dataset] = set()

    for _ in range(k + 1):
        next_layer = []

        for c in current_nodes:
            children = c.children.values()

            for c2 in children:
                next_layer.append(c2)
                d[c.dataset].add(c2.dataset)

        current_nodes = next_layer

    return d

def collapse_tree(
    root_node: TTNode,
    gin_layers: List[GINLayer],
    node_values: Dict[Dataset, torch.Tensor],
    children: Dict[Dataset, Set[Dataset]],
) -> torch.Tensor:


    return node_values[root_node.dataset]

