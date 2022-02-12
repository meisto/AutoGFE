#!/bin/python3
## ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 22 Dec 2021 01:35:39 AM CET
# Description: -
# ======================================================================
# stdlib imports
import json
import os
import random
import sys
import time

# Installed imports
import numpy as np

# Local imports
from src.env import ttenv
from src.util import score, experiment, dataset_loader
from src.util import lookup
from src.util.experiment import Experiment
from src.representation.v2 import representation_lookup
from src.representation import domain_index
from src.transformations import real, categorical, date, transformation
import src.agent.REINFORCE as REINFORCE
import src.agent.PPO as PPO

print(f"[INFO] Script started running at {time.asctime()}")




def main():
    # =========================================================================
    # =========================================================================
    # Extract paths
    DATA_PATH   = os.getenv("DATA_PATH")
    EXP_PATH    = os.getenv("EXP_PATH")
    PARAM_PATH  = os.getenv("PARAM_PATH")

    assert DATA_PATH != None
    assert EXP_PATH != None
    assert PARAM_PATH != None

    # =========================================================================
    # =========================================================================
    # Load datasets 
    all_datasets = dataset_loader.load_datasets(DATA_PATH)
    datasets = list(all_datasets)
    random.shuffle(datasets)

    # Filter out datasets from training because of bad performance.
    blacklist = ["default_credit", "churn_modelling", "spambase"]
    datasets = [x for x in datasets if x[0].name not in blacklist]

    print(f"[INFO] Loaded {len(datasets)} datasets from '{DATA_PATH}'.")

    # Shuffle and split datasets
    train_ratio = 0.8
    test_ratio  = 0.2

    assert int(train_ratio * len(datasets)) >= 1, "Not enough datasets."
    assert int(test_ratio * len(datasets)) >= 1, "Not enough datasets."

    i1 = int(train_ratio * len(datasets))
    train_datasets  = datasets[:i1]
    test_datasets   = datasets[i1:]



    # =========================================================================
    # =========================================================================
    # Load transformations
    transformations = real.all_transformations \
        .union(categorical.all_transformations) \
        .union(date.all_transformations)

    # Make sure there are not duplicates
    transformation.sanity_check(transformations)
    transformations = tuple(transformations)


    # =========================================================================
    # =========================================================================
    # Add transformations and datasets to lookup
    lookup.add_transformations(set(transformations))
    for ds in datasets: lookup.add_ds(ds[0])
    for ds in datasets: lookup.add_ds_value(ds[0], ds[1])

    # Add domain information
    domain_file         = os.path.join(DATA_PATH, "domains.json")
    print(f"[INFO] Load domain information from path '{domain_file}'.")
    domain_information  = {"domains": []}
    if os.path.exists(domain_file):
        with open(domain_file, mode='r') as f:
            domain_information = json.load(f)

    # Add domains to domain index
    ds_name = [x[0].name for x in datasets]
    for dom in domain_information["domains"]: 
        domain_index.add_domain(set(y for y in dom if y in ds_name))


    # =========================================================================
    # =========================================================================
    # Load all parameter sets
    params = next(os.walk(PARAM_PATH))[2]
    params = [x for x in params if x.endswith(".json")]
    params = [json.load(open(os.path.join(PARAM_PATH, x), mode='r')) for x in params]

    # Load all existing experiments
    experiments = next(os.walk(EXP_PATH))[1]

    
    params_names = [x["name"] for x in params]
    for x in experiments: assert x in params_names, f"Orphaned experiment '{x}'."

    params_wo_exp = [x for x in params if x["name"] not in experiments]


    print(f"[INFO] Loaded {len(params)} sets of parameters.")
    print(f"[INFO] Found experiments, ({len(params_wo_exp)}/{len(params)}) not run.")


    # =========================================================================
    # =========================================================================
    # Experiment

    if len(params_wo_exp) > 0:
        # Generate new experiment for an unexplored set of parameters

        current_params = params_wo_exp[0]

        exp = Experiment(
            train_datasets  = [x[0] for x in train_datasets],
            test_datasets   = [x[0] for x in test_datasets],
            transformations = transformations,
            parameters      = current_params,
            experiments_dir_path = EXP_PATH,
        )

    else:
        # Just terminate
        print("[WARNING] No free set of parameters, will terminate.")
        sys.exit()



    # =========================================================================
    # =========================================================================
    # Hyperparameters
    TRAIN_DATASETS                  = train_datasets
    TEST_DATASETS                   = test_datasets       


    EVALUATION_FUNCTION             = lambda x,c: score.evaluate(
        dataset=x,
        context=c,
        modeltype=exp.parameters["evaluation_model"]
    )

    REPRESENTATION_GENERATOR    = representation_lookup(
        exp.parameters["representation_generator"],
        exp.parameters["representation_dimension"],
    )

    # Environments
    ENVIRONMENT_PARAMETERS = {
        "representation_dimension"  : exp.parameters["representation_dimension"],
        "representation_generator"  : REPRESENTATION_GENERATOR,
        "evaluation_function"       : EVALUATION_FUNCTION,
        "max_steps"                 : exp.parameters["max_number_steps"],
        "reward_function"           : exp.parameters["reward_function"],
    }


    # =========================================================================
    # =========================================================================
    # Environments
    generate_environment = lambda dataset, feature_values: \
        ttenv.TTEnv(
            root_dataset                = dataset,
            feature_values              = feature_values,
            **ENVIRONMENT_PARAMETERS
        ) 
 
    print("[INFO] Start building testing environments.")
    test_envs = [generate_environment(ds, fv) for ds, fv in test_datasets]

    print("[INFO] Start building training environments.")
    train_envs = [generate_environment(ds, fv) for ds, fv in train_datasets]

    # =========================================================================
    # =========================================================================
    # Do the actual training
    get_name = lambda x: x[0].name if x[0].name != None else "noname"
    print(f"[INFO] Training datasets:")
    print("   " + ", ".join(get_name(x) for x in TRAIN_DATASETS))

    print(f"[INFO] Testing datasets:")
    print("   " + ", ".join(get_name(x) for x in TEST_DATASETS))

    print("[INFO] Finished setup. Start training.\n\n")

    print(f"[INFO] Parameter set: {exp.parameters['name']}")

    # Start training
    if exp.parameters["learning_algorithm"] == "REINFORCE":
        REINFORCE.run(
            experiment          = exp,
            logging_interval    = lambda x: x % 25 == 0,
            train_envs          = train_envs,
            test_envs           = test_envs,
        )
    elif exp.parameters["learning_algorithm"] == "PPO":
        PPO.run(
            experiment          = exp,
            logging_interval    = lambda x: x % 25 == 0,
            train_envs          = train_envs,
            test_envs           = test_envs,
        )

if __name__ == "__main__":
    main()
