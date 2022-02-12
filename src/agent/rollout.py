# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Tue 08 Feb 2022 09:44:34 PM CET
# Description: -
# ======================================================================
# stdlib imports
from typing import List, Callable, Dict, Optional
from dataclasses import dataclass, field
import time

# library imports
import numpy as np
import torch

# local file imports
from src.env.feature import Transformation
from src.env.ttenv import TTEnv
from src.env.ttree import TTNode
from src.util.experiment import Experiment
from src.env.feature import Dataset, Feature
from src.representation.v2 import representation_lookup
from src.representation.major import Major


@dataclass(frozen=False)
class RolloutBuffer:
    actions: List[torch.Tensor]         = field(default_factory=lambda: list())
    nodes: List[np.ndarray]             = field(default_factory=lambda: list())
    majors: List[Major]                 = field(default_factory=lambda: list())
    rewards: List[float]                = field(default_factory=lambda: list())
    performances: List[float]           = field(default_factory=lambda: list())
    logprobs: List                      = field(default_factory=lambda: list())

    
    def clear(self):
        del self.actions[:]
        del self.nodes[:]
        del self.majors[:]
        del self.rewards[:]
        del self.performances[:]
        del self.logprobs[:]

    def append(self, other_buffer):
        """
            Add one rollout buffer to another.
        """
        self.actions.extend(other_buffer.actions)
        self.nodes.extend(other_buffer.nodes)
        self.majors.extend(other_buffer.majors)
        self.rewards.extend(other_buffer.rewards)
        self.performances.extend(other_buffer.performances)
        self.logprobs.extend(other_buffer.performances)

    def discount_rewards(self, gamma):
        """
            Discount rewards in this buffer.
        """

        discounted = self.rewards.copy()
        for i in reversed(range(len(self.rewards) - 1)):
            discounted[i] = discounted[i] + gamma * discounted[i + 1]

        self.rewards = discounted
def _action_to_transformation(x: int, experiment: Experiment) -> Optional[Transformation]:
    """ 
        Helper function. Transform an integer x into a transformation
        or None (a reduction) according to value.
    """

    if x == (experiment.num_transformations - 1):
        return None
    else:
        return experiment.transformations[x]

def transform_dataset(
    root_dataset: Dataset,
    feature_values: Dict[Feature, np.ndarray],
    evaluation_function: Callable,
    experiment: Experiment,
    predict: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    greedy: bool = False,
) -> RolloutBuffer:
    """
        Generate a single rollout.
    """
 
    ## Generate environment
    representation_generator = representation_lookup(
        experiment.parameters["representation_generator"],
        experiment.parameters["representation_dimension"]
    )

    env_params = {
        "representation_dimension": experiment.parameters["representation_dimension"],
        "representation_generator": representation_generator,
        "evaluation_function":      evaluation_function,
        "max_steps": experiment.parameters["max_number_steps"],
        "reward_function": experiment.parameters["reward_function"],
    }

    env = TTEnv(
        root_dataset    = root_dataset,
        feature_values  = feature_values,
        **env_params,
    )
    
    # Generate a rollout on the new env
    generate_rollout(env, experiment)

#    policy = experiment.policy
#
#    # Reset variables for new rollout
#    state = env.reset()
#    done = False
#    while not done: # Episode
#
#        # Convert state (dataset) into dataset representation
#        representation = env.major.get_dataset_representation(state)
#
#        # Make prediction, choose action and take action
#        prediction = predict(policy, representation).detach()
#
#        # Sample randomly
#        if not greedy:
#            dist    = torch.distributions.Categorical(prediction)
#            action  = dist.sample()
#        else:
#            action = prediction.argmax()
#        
#        # Take step in environment
#        x = _action_to_transformation(action.item(), experiment)
#        state, _, done, _ = env.step(x)

    return env.best_performance

def generate_rollout(
    env: TTEnv,
    experiment: Experiment,
) -> RolloutBuffer:
    """
        Generate a single rollout.
    """
    t1 = time.time()
    agent = experiment.policy
    def _action_to_transformation(x: int) -> Optional[Transformation]:
        """ 
            Helper function. Transform an integer x into a transformation
            or None (a reduction) according to value.
        """

        if x == (experiment.num_transformations - 1):
            return None
        else:
            return experiment.transformations[x]

    # Save results
    buffer = RolloutBuffer()

    # Reset variables for new rollout
    state = env.reset()
    done = False

    while not done: # Episode
        if experiment.parameters["learning_algorithm"] == "REINFORCE":
            # Convert state (dataset) into prediction
            prediction = agent.forward(state, env.major)
        else:
            # Convert state (dataset) into prediction, discard critic
            prediction, _ = agent.forward(state, env.major)

        # Sample randomly
        dist    = torch.distributions.Categorical(prediction)
        action  = dist.sample()
        
        # Save logprob so we must not recalculate
        logprob = dist.log_prob(action)

        # Take step in environment
        x = _action_to_transformation(action.item())
        state, reward, done, info = env.step(x)

        # Save results in traces
        # buffer.representations.append(env.major.get_dataset_representation(state.dataset))
        buffer.nodes.append(state)
        buffer.majors.append(env.major)
        buffer.performances.append(info["performance"])
        buffer.actions.append(action.detach())
        buffer.rewards.append(reward)

        # Save logprobs, not actually use in REINFORCE but needed for PPO
        buffer.logprobs.append(logprob)   

    t2 = time.time()

    if t2 - t1 > 30:
        print(f"[WARNING] Dataset {env.root_dataset.name} performed badly ({t2-t1}) seconds).")
    return buffer

def evaluate(
    experiment: Experiment,
    envs: List[TTEnv],
    number_runs: int,
) -> List[float]:
    """
        Given a policy, a list of environments and a number of runs, evaluate
        the policy and return for each environment the resulting boost in 
        performance.
    """
    results = []

    # Iterate through all environments
    for env in envs:
        # Protocol performance in the environment
        env_performances = []

        # Iterate a number of times over each feature
        for _ in range(number_runs):

            # Generate a roolout for each environment
            res = generate_rollout(env, experiment)

            # Subtract base performance from performances
            performances = res.performances
            root_performance = env.root_performance
            for i,p in enumerate(performances):
                performances[i] = p - root_performance
            
            env_performances.append(max(performances))

        results.append(sum(env_performances) / len(env_performances))

    return results
