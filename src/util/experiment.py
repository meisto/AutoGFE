# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Thu 06 Jan 2022 06:10:44 PM CET
# Description: -
# ======================================================================
# stdlib import
import json
import os
import pickle
import time
from typing import List, Dict, Any, Tuple, Optional

# downloaded imports
import pandas as pd
import torch

# local imports
from src.agent.generate_policy import generate_agent
from src.env.feature import Dataset, Transformation


class Experiment:
    def __init__(
        self,
        train_datasets: List[Dataset],
        test_datasets: List[Dataset],
        transformations: Tuple[Transformation,...],
        parameters: Dict[str, Any],
        experiments_dir_path: str,
    ):
        self.initial_timestamp = self.generate_timestamp()
        self.current_timestep = 0

        # experiment is done
        self.done = False

        # Try to generate own directory
        self.dir_path = os.path.join(experiments_dir_path, f"{parameters['name']}")

        # Generate a name for this experiment
        self.name = str(hash(self.initial_timestamp))
        
        self.exp_file_name = f"{self.dir_path}{self.name}.exp"

        # Make dir if it does not exist 
        if not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)
        
        # Check if name already exists
        assert not os.path.exists(self.exp_file_name),\
            "Experiment already exists."


        # Generate log datastructures
        self.losses             = []
        self.sum_rewards        = []
        self.mean_rewards       = []
        self.train_performances = []
        self.test_performances  = []

        # Set hyperparameters
        self.transformations    = transformations
        self.training_datasets  = [x.name for x in train_datasets]
        self.testing_datasets   = [x.name for x in test_datasets]
        self.parameters         = parameters

        # Some parameters need updates depending on runtime
        self.num_transformations = len(self.transformations) + 1 # +1 for reduction

        # Generate policy
        self.policy: torch.nn.Module = generate_agent(
            self.parameters,
            self.num_transformations
        )

    def generate_timestamp(self) -> str:
        return time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())


    ##
    ## Loading and saving capabilities.
    ##

    def generate_savepoint(
        self,
        train_performance: float,
        test_performance: float,
    ):
        # Generate file paths
        specific_name = self.name   # "_s{str(timestep).zfill(4)}"
        policy_file_name_sd = os.path.join(self.dir_path, specific_name + ".sd")
        exp_file_name       = os.path.join(self.dir_path, specific_name + ".exp")


        #NOTE: ONNX export not used, might be used later.
        # policy_file_name_onxx   = f"{self.root_path}model{specific_name}.onnx" 
        # torch.onnx.export(
        #     model           = policy,
        #     args            = {},
        #     f               = policy_file_name_onxx,
        #     input_names     = ["input1"],
        #     output_names    = ["output1"],
        # )

        # Write policy
        torch.save(self.policy.state_dict(), policy_file_name_sd)

        # Write a copy of the policy for later restoration
        policy_backup_path = os.path.join(
            self.dir_path,
            specific_name + f"step_{str(self.current_timestep).zfill(4)}.sd"
        )
        torch.save(self.policy.state_dict(), policy_backup_path)

        results_path = os.path.join(
            self.dir_path,
            specific_name + f"step_{str(self.current_timestep).zfill(4)}.json"
        )
        json.dump(
            {
                "train_performance": train_performance,
                "test_performance": test_performance
            },
            open(results_path, mode='w')
        )

        policy = self.policy
        self.policy = None

        # Write experiment
        with open(exp_file_name, mode='wb') as f: pickle.dump(self, f)

        self.policy = policy

    def load_agent_parameters(self, parameter_path):
        # print(f"[INFO] Load parameters from path '{parameter_path}'.")

        # Generate policy and load values
        self.policy = generate_agent(self.parameters, self.num_transformations)
        self.policy.load_state_dict(torch.load(parameter_path)) 


def load_savepoint(dir_path) -> Experiment:


    found_files = next(os.walk(dir_path))[2]

    # found_files_csv = [x for x in found_files if x.endswith(".csv")]
    found_files_sd  = [x for x in found_files if x.endswith(".sd")]
    found_files_exp = [x for x in found_files if x.endswith(".exp")]

    # Check if both .csv and .sd files exist
    bases = sorted([x[:-3] for x in found_files_sd])
    # assert all([x + ".csv" in found_files_csv for x in bases]), "File missing."
    assert all([x + ".sd" in found_files_sd for x in bases]), "File missing."
    assert len(found_files_exp) == 1, "Too many experiment files."

    selection = bases[0]

    # Get paths
    path_exp    = os.path.join(dir_path, selection + ".exp")
    path_sd     = os.path.join(dir_path, selection + ".sd")
    # path_csv    = os.path.join(dir_path, selection + ".csv")

    # Load experiment
    exp = pickle.load(open(path_exp, mode='rb'))

    # Generate policy and load values
    # exp.policy = generate_agent(exp.parameters, exp.num_transformations)
    # exp.policy.load_state_dict(torch.load(path_sd)) 

    return exp
