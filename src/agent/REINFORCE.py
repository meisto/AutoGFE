# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Thu 23 Dec 2021 12:10:48 AM CET
# Description: -
# ======================================================================
import random
import time
import traceback
from typing import List, Callable

import torch

from src.env.ttenv import TTEnv
from src.util.experiment import Experiment
from src.agent.rollout import RolloutBuffer, generate_rollout, evaluate


def run(
    experiment: Experiment,
    logging_interval: Callable[[int], bool],
    train_envs: List[TTEnv],
    test_envs: List[TTEnv],
):
    print(f"[NOTE] Started training at {time.asctime()}")
    assert experiment.parameters["learning_algorithm"] == "REINFORCE"

    # Extract hyperparameters
    gamma               = experiment.parameters["gamma"]
    num_trajectories    = experiment.parameters["num_trajectories"]
    num_iterations      = experiment.parameters["num_iterations"]
    num_train_steps     = experiment.parameters["num_train_steps"]

    # Generate optimizer
    optimizer = torch.optim.Adam(experiment.policy.parameters(), lr=0.0005)

    # Other helpers
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

        
            rewards  = torch.tensor(episode_buffer.rewards).float()
            actions = torch.tensor(episode_buffer.actions).detach()

            experiment.sum_rewards.append(rewards.sum().item())
            experiment.mean_rewards.append(rewards.mean().item())

            # Normalize rewards
            rewards  = (rewards - rewards.mean()) / (rewards.std() + 10e-6)


            # Optimize policy num_train_steps times
            for _ in range(num_train_steps):

                # Zero out optimizer
                optimizer.zero_grad()

                # Generate forward pass over the whole batch
                probs = experiment.policy.forward_batch(
                    episode_buffer.nodes,
                    episode_buffer.majors,
                )

                # Build loss
                gathered = torch.gather(probs, -1, actions.unsqueeze(dim=1).detach()).squeeze()

                # 
                loss = -(gathered.log() * rewards).mean()

                # Optimize over loss
                loss.backward()
                optimizer.step()

            # Log rewards and losses
            experiment.losses.append(loss.item())


            # In certain intervall evaluate on testing dataset
            if logging_interval(ep_no) or ep_no == (num_iterations - 1):
                print(f"[NOTE] Saved model '{experiment.name}' at timestep {ep_no}.")
                train_performance = evaluate(
                    experiment  = experiment,
                    envs        = random.sample(train_envs, k=5),
                    number_runs = 3,
                )
                train_performance = sum(train_performance) / len(train_performance)

                test_performance = evaluate(
                    experiment  = experiment,
                    envs        = random.sample(test_envs, k=5),
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
                f"Mean reward: {experiment.mean_rewards[-1]:>6.4f}"
            )
            # If code reaches here: reset
            error_counter = 0
        except:
            print("[ERROR] Error occured during training")
            traceback.print_exc()
            error_counter += 1
        ## Bookmark: End of episode ##

    # Experiment done
