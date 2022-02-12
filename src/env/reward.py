# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Sun 09 Jan 2022 03:46:35 AM CET
# Description: -
# ======================================================================
from typing import Callable

def binary_reward(
    current_performance: float,
    previous_performance: float,
    best_performance: float, # best_performance
) -> float:

    if current_performance > previous_performance:
        return 1.0
    else:
        return -1.0
    
def staggered_reward(
    current_performance: float,
    previous_performance: float,
    best_performance: float,
) -> float:

    if current_performance == 1:
        # Achieved a perfect results
        reward = 10.0    # Very good results
    # elif performance > self.best_performance:
    #     reward = 2.0 + performance
    elif current_performance > previous_performance:
        reward = 1.0 + current_performance
    else:
        # Else give gain in performance as reward
        reward = 0.0

    return reward

    
def get_reward_function(choice: str) -> Callable: 
    choices = ["binary_reward", "staggered_reward"]
    assert choice in choices, f"Choice '{choice}' not a known reward function."

    if choice == "binary_reward":
        return binary_reward
    elif choice == "staggered_reward":
        return staggered_reward

    raise Exception("Exception")
