# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 22 Dec 2021 09:54:59 PM CET
# Description: -
# ======================================================================
from typing import Tuple, Set, Dict
from collections import defaultdict

import numpy    as np
import torch

from src.representation import domain_index
from src.env.ttree import TTNode
from src.env.feature import Dataset
from src.representation.major import Major
from src.util import lookup

from .collapse_tree import get_relevant_datasets
from .gin_layer import GINLayer
from .random_tree import create_random_tree

class REINFORCEAgent(torch.nn.Module):
    def __init__(
        self,
        fe_arch: Tuple[int, ...],
        prob_arch: Tuple[int, ...],
        num_transforms: int,
        enable_local_context = False,
        enable_domain_information = False,
    ):
        super(REINFORCEAgent, self).__init__()

        assert len(fe_arch) > 1, "Illegal feature extractor architecture."
        assert len(fe_arch) > 0, "Illegal probability extractor architecture."

        # Safe Metaparameters
        self.representation_dimension = fe_arch[0]

        # Generate feature extractor
        fe = []
        for i in range(len(fe_arch) - 1):
            fe.append(torch.nn.Linear(fe_arch[i], fe_arch[i+1]))
            fe.append(torch.nn.ReLU())
        self.feature_extractor: torch.nn.Sequential = \
            torch.nn.Sequential(*fe)
        
        # Aggregator
        self.aggregator = lambda x: x.mean(dim=0) 

        # Generate probability extractor
        prob = []
        for i in range(len(prob_arch) - 1):
            prob.append(torch.nn.Linear(prob_arch[i], prob_arch[i+1]))
            prob.append(torch.nn.ReLU())
        prob.append(torch.nn.Linear(prob_arch[-1], num_transforms))
        prob.append(torch.nn.ReLU())
        prob.append(torch.nn.Softmax(dim=-1))
        self.prob_extractor: torch.nn.Sequential = \
            torch.nn.Sequential(*prob)

        #
        self.enable_local_context = enable_local_context
        self.enable_domain_information = enable_domain_information

        self.context_gin_layers = [
            GINLayer(1.1, prob_arch[0]) for _ in range(4)
        ]

        self.domain_gin_layers = [
            GINLayer(1.1, prob_arch[0]) for _ in range(4)
        ]
        self.domain_reducer = torch.nn.Linear(2 * prob_arch[0], prob_arch[0])





    def get_dataset_representation(
            self,
            datasets: Tuple[Dataset, ...],
            majors: Tuple[Major, ...],
            from_lookup: bool = False
    ) -> torch.Tensor:
        """
            Given a dataset and an object that represents the context,
            return a dataset reprentation.
        """
        assert len(datasets) > 0
        assert len(majors) == len(datasets)

        # Get raw feature representations
        representations = tuple(major.get_dataset_representation(
            dataset, 
            from_lookup,
            dataset.name
        ) for major, dataset in zip(majors, datasets))

        # Get learned representations
        return self.get_dataset_representation_already_extracted(
            representations
        )

    def get_dataset_representation_already_extracted(
            self,
            representations: Tuple[np.ndarray, ...],
        ) -> torch.Tensor:
        
        normalized = []
        for x in representations:
            # assert len(x.shape) == 2, f"All must be 2D, is '{len(x.shape)}'."
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=0)

            normalized.append(x)
        representations = tuple(normalized)

        assert all(len(x.shape) == 2 for x in representations)

        # Concatenate all representations and generate features
        bulk        = np.concatenate(representations, axis=0)
        bulk        = torch.tensor(bulk).float()
        
        # Add additional dimension for inputs with only one
        if len(bulk.shape) == 1:
            bulk = bulk.unsqueeze(dim=0)

        # Create feature representations
        features    = self.feature_extractor(bulk)

        assert len(features.shape) == 2, f"{features.shape}"

            
        # Split the data back into individual datasets
        split = []
        count = 0
        for y in representations:
            x = y.shape
            
            # Cut the old piece out and aggregate it
            split.append(self.aggregator(features[count:count + x[0],:]))

            # Update count
            count += x[0]


        features  = torch.stack(split, dim=0)

        # Sanity checks
        assert features.shape[0] == len(representations), \
            f"{features.shape[0]} != {len(representations)}"

        return features
    
    def forward(self, current_node: TTNode, major) -> torch.Tensor:
        assert type(current_node) == TTNode
        assert type(major) == Major

        return self.forward_batch((current_node,), (major,))[0,:]

    def forward_batch(
        self,
        current_nodes: Tuple[TTNode, ...],
        majors: Tuple[Major, ...],
    ) -> torch.Tensor:
        assert len(current_nodes) == len(majors), f"{len(current_nodes)} != {len(majors)}" 

        current_datasets = [n.dataset for n in current_nodes]

        if self.enable_local_context:

            children = {}
            children_to_major = {}
            # Extract graph structure
            for current_node, major in zip(current_nodes, majors):
                new = get_relevant_datasets(
                    current_node,
                    len(self.context_gin_layers)
                )

                # Save major for each
                for c in new.values():
                    for x in c: children_to_major[x] = major
                for k in new.keys(): children_to_major[k] = major

                children.update(new)

            # Get all relevant nodes
            all_datasets = set(children.keys())
            for k in children.keys():
                all_datasets.update(children[k])
            all_datasets = tuple(all_datasets)

            # Generate initial representations
            ms = tuple(children_to_major[x] for x in all_datasets)
            representations = self.get_dataset_representation(all_datasets, ms)

            # Generate representation for all relevant nodes.
            dataset_representations = {}
            for i, ds in enumerate(all_datasets):
                dataset_representations[ds] = representations[i,:]

            # Iterate through GIN layers
            for gl in self.context_gin_layers:
                # 
                dataset_representations = gl.forward(
                    children, 
                    dataset_representations,
                )

            # Extract representation of root from the tree
            x = tuple(dataset_representations[cn.dataset] for cn in current_nodes)
            x = torch.stack(x, dim=0)

        else:
            x = self.get_dataset_representation(tuple(current_datasets), majors)


        if self.enable_domain_information:
            # Add domain informations

            # Load transformations
            transformations = lookup.get_all_transformations()

            all_domain_members = [
                domain_index.get_other_domain_members(cd.dataset.name) 
                    for cd in current_nodes
            ]

            children: Dict[Dataset, Set[Dataset]] = defaultdict(lambda: set())

            root_nodes = []
            initial_representations: Dict[Dataset, np.ndarray] = {}

            for domain_members in all_domain_members:
                domain_root_nodes = []

                for dm in domain_members:
                    # Get dataset and context
                    ds, context = lookup.get_ds(dm)

                    # Create a random tree 
                    domain_tree = create_random_tree(
                        ds,
                        transformations, 
                        32,
                        context
                    )
                    domain_root_node = domain_tree.root_node
                    domain_root_nodes.append(domain_root_node)

                    already_visited = set()
                    todo = [domain_root_node]
                    while len(todo) > 0:
                        current = todo.pop()

                        # Get all children of the node
                        ch = []
                        for c in current.children.values():
                            if c.dataset not in already_visited:
                                ch.append(c)
                        todo.extend(ch)

                        # Add to tree structure
                        for c in ch:
                            children[current.dataset].add(c.dataset)

                        # Add node to children
                        already_visited.add(current.dataset)
                    
                    for ds in already_visited:
                        # Update initial representation
                        y = np.array([context[x] for x in ds.features])
                        initial_representations[ds] = y

                # Add these nodes to the global
                root_nodes.append(domain_root_nodes)

            # Get all relevant nodes
            all_datasets = set()
            all_datasets.update(children.keys())
            for k in children.keys():
                all_datasets.update(children[k])
            all_datasets = tuple(all_datasets)

            # Need no gradient here
            with torch.no_grad():
                x = self.get_dataset_representation_already_extracted(
                    tuple(initial_representations[x] for x in all_datasets)
                )

            # Generate representation for all relevant nodes.
            dataset_representations: Dict = {}
            for i, ds in enumerate(all_datasets):
                dataset_representations[ds] = x[i,:]

            # Iterate through GIN layers
            for gl in self.domain_gin_layers:
                # 
                dataset_representations = gl.forward(
                    children, 
                    dataset_representations,
                )

            # Extract representation of root from the tree
            dom_res = []
            for l1 in root_nodes:
                dom_reps = [
                    dataset_representations[l2] for l2 in l1
                ]

                # In case of no domain info: use zero vector
                if len(dom_reps) == 0: 
                    dom_reps.append(torch.zeros(self.representation_dimension))
    
                # Aggregate domain tree representations
                dom_res.append(torch.stack(dom_reps,dim=0).mean(dim=0))

            #   
            dom_res = torch.stack(dom_res, dim=0)
            
            # Concatenate the new representation and the dataset representation
            x = torch.cat([x,dom_res], dim=-1)

            # Reduce to normal size
            x = self.domain_reducer(x)




        # Extract probabilites
        x = self.prob_extractor(x)

        if x.isnan().any():
            print("[ERROR] Prediction is nan for at least one value in MLP.")
            print(x)
            raise Exception("Illegal value in prediction.")
        return x

