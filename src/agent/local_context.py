# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Thu 10 Feb 2022 08:20:02 PM CET
# Description: -
# ======================================================================

# library imports
import torch

# local file imports
from .gin_layer import GINLayer

class DomainInformation(torch.nn.Module):
    
    def __init_(
        self,
        num_cov_layers: int,
        gin_epsilon: float,
        dimensionality: int,
    ):
        super(DomainInformation, self).__init__()


        gin_layers = [
            GINLayer(gin_epsilon, dimensionality) 
            for _ in range(num_cov_layers)
        ]
        self.gin_layers = torch.nn.Sequential(
            *gin_layers
        )

