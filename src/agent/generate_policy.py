# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Tue 08 Feb 2022 11:14:15 PM CET
# Description: -
# ======================================================================
from typing import Dict

import torch

from src.agent.REINFORCE_agent import REINFORCEAgent
from src.agent.PPO_agent import PPOAgent

def generate_agent(parameters: Dict, num_transforms: int) -> torch.nn.Module:
    # Load policy


    la = parameters["learning_algorithm"]

    if la == "REINFORCE":
        agent = REINFORCEAgent(
            num_transforms=num_transforms,
            **parameters["policy_parameters"]
        )
    elif la == "PPO":
        agent = PPOAgent(
            num_transforms=num_transforms,
            **parameters["policy_parameters"]
        )

    else:
        raise Exception("Illegal state exception")

    return agent



