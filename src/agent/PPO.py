# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Fri 28 Jan 2022 03:39:47 PM CET
# Description: -
# ======================================================================
# stdlib imports
import random
import time
import traceback
from typing import List, Callable

# library imports
import torch

# local file imports
from src.util.experiment import Experiment
from src.agent.rollout import RolloutBuffer, generate_rollout, evaluate
from src.env.ttenv import TTEnv

# TODO: Cross-check with paper
# Set current policy and old policy to be equal
# self.policy = ActorCritic(state_dim, action_dim)
# self.policy_old = ActorCritic(state_dim, action_dim)
# self.policy_old.load_state_dict(self.policy.state_dict())



def run(
    experiment: Experiment,
    logging_interval: Callable[[int], bool],
    train_envs: List[TTEnv],
    test_envs: List[TTEnv],
):
    """ 
        Update the policy network based on the values in the rollout 
        buffer.
    """
    print(f"[NOTE] Started training at {time.asctime()}")
    assert experiment.parameters["learning_algorithm"] == "PPO"

    # Extract hyperparameters
    gamma               = experiment.parameters["gamma"]
    epsilon_clip        = experiment.parameters["epsilon_clip"]
    num_trajectories    = experiment.parameters["num_trajectories"]
    num_iterations      = experiment.parameters["num_iterations"]
    num_train_steps     = experiment.parameters["num_train_steps"]

    # Generate optimizer
    # optimizer = torch.optim.Adam([
    #     {'params': experiment.policy.actor.parameters(), 'lr': 0.0005},
    #     {'params': experiment.policy.critic.parameters(), 'lr': 0.0005}
    # ])
    optimizer = torch.optim.Adam(experiment.policy.parameters(), lr=0.0005)

    # Other helpers
    mse = torch.nn.MSELoss()
    episode_buffer = RolloutBuffer()
    current_timestep = experiment.current_timestep

    error_counter = 0
    for ep_no in range(current_timestep, num_iterations):       
        ## Bookmark: Start of episode ##
        try:

            if error_counter > 5:
                print("[ERROR] To many errors in sequence during loop.")
                break;

            # Reset buffer
            episode_buffer.clear()

            # Generate rollouts
            for env in random.choices(train_envs, k=num_trajectories):
                # Iterate through random subsamples of envs

                # Generate rollout
                buffer = generate_rollout(env, experiment)

                # Discount buffer
                buffer.discount_rewards(gamma)

                # Add buffer to episode buffer
                episode_buffer.append(buffer)

            # Normalize rewards
            rewards = torch.tensor(episode_buffer.rewards).float()

            experiment.sum_rewards.append(rewards.sum().item())
            experiment.mean_rewards.append(rewards.mean().item())

            rewards = (rewards - rewards.mean()) / (rewards.std() + 10e-6)

            # convert list to tensor
            old_actions  = torch.stack(episode_buffer.actions, dim=0).detach()
            old_logprobs = torch.tensor(episode_buffer.logprobs).detach()

            # Optimize policy num_train_steps times
            for _ in range(num_train_steps):

                # Zero out optimizer
                optimizer.zero_grad()

                # Generate new logprobs, values and dist_entropy
                new_predictions, new_values = experiment.policy.forward_batch(
                    episode_buffer.nodes,
                    episode_buffer.majors,
                )

                # Generate a distribution
                dist = torch.distributions.Categorical(new_predictions)

                # Generate logprobs and entropy
                new_logprobs    = dist.log_prob(old_actions.detach())
                new_entropy     = dist.entropy()

                # Calculate loss
                ratios:     torch.Tensor = torch.exp(new_logprobs - old_logprobs)
                advantages: torch.Tensor = rewards - new_values.detach()   
                clamped:    torch.Tensor = torch.clamp(
                    ratios,
                    1.0 - epsilon_clip,
                    1.0 + epsilon_clip
                )
                loss =\
                    -torch.min(ratios * advantages, clamped * advantages)\
                    + 0.5 * mse(new_values, rewards)\
                    - 0.01 * new_entropy
                loss = loss.mean()
                
                # Optimize over loss
                loss.backward()
                optimizer.step()
            
            # Log rewards and losses
            experiment.losses.append(loss.item())



            # If required save the model to disk
            if logging_interval(ep_no) or ep_no == (num_iterations - 1):
                print(f"[NOTE] Saving model '{experiment.name}' at timestep {ep_no}.")
                train_performance = evaluate(
                    experiment  = experiment,
                    envs        = train_envs,
                    number_runs = 3,
                )
                train_performance = sum(train_performance) / len(train_performance)

                test_performance = evaluate(
                    experiment  = experiment,
                    envs        = test_envs,
                    number_runs = 3,
                )
                test_performance = sum(test_performance) / len(test_performance)

                # Log performances
                experiment.train_performances.append(train_performance)
                experiment.test_performances.append(test_performance)

                # Update and save experiment for recovery
                experiment.current_timestep = ep_no
                experiment.generate_savepoint(
                    train_performance=train_performance,
                    test_performance=test_performance
                )
                print(f"[NOTE] Saved model '{experiment.name}' at timestep {ep_no}.")

            print(
                f"[TRAINING] Step: {ep_no:>2} | "
                f"Mean loss: {experiment.losses[-1]:>9.6f} | "
                f"Mean reward: {experiment.mean_rewards[-1]:>6.4f} | "
            )

            # If code reaches here: reset
            error_counter = 0
        except:
            print("[ERROR] Error occured during training")
            traceback.print_exc()
            error_counter += 1

    # Experiment done
