# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 29 Dec 2021 04:06:14 PM CET
# Description: -
# ======================================================================
from typing import Tuple, Dict, Optional
import random 
import torch

from src.env.feature import Feature, Dataset


def discounted_rewards(
    rewards: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    l = rewards.shape[-1]

    disc_rew = torch.zeros_like(rewards)
    disc_rew[l-1] = rewards[l-1]
    
    for i in reversed(range(l - 1)):
        disc_rew[i] = rewards[i] + gamma * disc_rew[i+1]

    return disc_rew

def epsilon_greedy(
    prediction: torch.Tensor,
    epsilon: float,
    ds: Dataset
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[Feature, float]]]:

    if prediction.isnan().any():
        print("[ERROR] Prediction is nan for at least one value.")
        print(prediction)
        raise Exception("Illegal value in prediction.")

    if len(prediction.shape) == 1:
        if random.random() <= epsilon:
            # Random sample
            prob = torch.distributions.Categorical(prediction)
            action = prob.sample()
        else:
            # Greedy
            action = prediction.argmax()

        res_prob = None

    else:
        maxes = prediction.max(dim=0)[0]
        maxes = maxes + 10e-5
        maxes = torch.softmax(maxes, dim=0)

        action = torch.distributions.Categorical(maxes).sample()

        prob_tensor = prediction[:,action]
        prediction = maxes

        res_prob = {ds.features[i]: float(prob_tensor[i].item())
                for i in range(prob_tensor.shape[0])}

    return (prediction, action, res_prob)
