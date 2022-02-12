# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Thu 10 Feb 2022 12:07:11 AM CET
# Description: -
# ======================================================================
# stdlib imports
from typing import FrozenSet, Dict, Callable, Optional, Set, Tuple, List
import random
import time
import warnings
import json
import sys
import os

# library imports
import numpy as np

# Local file imports
from src.env.feature import Dataset, Transformation, Feature
from src.env.ttree import TTree
from src.transformations import real, transformation
from src.util import score
from src.util.dataset_loader import load_datasets

def recog(
    root_dataset: Dataset,
    transformations: FrozenSet[Transformation],
    number_expansions: int,
    feature_values: Dict[Feature, np.ndarray],
    evaluation_function: Callable[[Dataset, Dict[Feature, np.ndarray]], float],
) -> Tuple[float, float]:
    """
    Reimplementation of the cognito algorithm as discusses in the papter.
    """
    labels = root_dataset.labels
    labels_values = feature_values[labels]

    def step(
        pairs: List[Tuple[Feature, np.ndarray]],
        transformation: Transformation,
        label: Feature,
    ) -> bool:

        possible_inputs = transformation.get_possible_inputs(Dataset(
            frozenset(x[0] for x in pairs).union(frozenset([label])), label))

        if len(possible_inputs) == 0: return False
        possible_input = random.sample(possible_inputs, 1)[0]

        context = {x[0]: x[1] for x in pairs}
        new_pair = transformation.apply_wo_context(possible_input, context)
        discard_pair = random.sample(pairs, 1)[0]

        pairs.append(new_pair)
        pairs.remove(discard_pair)

        return True
    # Hyperparameter
    MAX_NUMBER_GAMMA = 5

    def evaluate(p):
        tmp = {x[0]: x[1] for x in p}
        tmp[labels] = labels_values
        return evaluation_function(
            Dataset(
                frozenset(x[0] for x in p),
                root_dataset.labels
            ),
            tmp
        )


    current_pairs           = list((x, feature_values[x]) for x in root_dataset.features)
    current_performance     = evaluate(current_pairs)


    root_performance = current_performance

    best_performance    = current_performance
    best_pairs          = current_pairs

    # Sample a random initial transformation
    transform = random.sample(transformations, 1)[0]

    gamma_counter = 0
    for _ in range(number_expansions):

        # Update
        is_good = step(current_pairs, transform, root_dataset.labels)
        new_performance = evaluate(current_pairs)

        if not is_good:
            # No transformations possible anymore => go to best result
            del(current_pairs)
            current_pairs       = best_pairs.copy()
            current_performance = best_performance

            # Sample a new transformation
            transform = random.sample(transformations, 1)[0]

            
        else:
            if new_performance > current_performance:
                gamma_counter = 0

                current_performance = new_performance
                
                # Update best performance
                if new_performance > best_performance:
                    del(best_performance)
                    best_performance    = new_performance
                    best_pairs          = current_pairs.copy()
                

            else:
                gamma_counter += 1
                
                if gamma_counter >= MAX_NUMBER_GAMMA:
                    # Sample a new transformation
                    transform = random.sample(transformations, 1)[0]
                    del(current_pairs)
                    current_pairs           = best_pairs.copy()
                    current_performance     = best_performance

    return (best_performance, root_performance)

if __name__ == "__main__":
    all_datasets = load_datasets("/home/tobias/ma/data")
    all_datasets = {x[0].name: x for x in all_datasets}
    
    transformations = real.all_transformations
    transformation.sanity_check(transformations)


    test_datasets = ["default_credit", "diabetes", "spect_heart"]
    test_datasets = [x for x in test_datasets if x in all_datasets]
    random.shuffle(test_datasets)

    # Select a file without test results
    ds_name = None
    path = ""
    for t in test_datasets:
        path = os.path.join("/home/tobias/ma/", t)

        if not os.path.exists(path): 
            ds_name = t
            fn= path
            
            # Create file to prevent duplicates.
            with open(path, mode='w') as f:
                f.write(" ")
                f.flush()

            break

    if ds_name == None:
        print("[ERROR] Results for all datasets already exist, terminating")
        sys.exit()

    print(f"[INFO] Start generating recog results for dataset '{ds_name}'.")

    all_results = dict()
    results = []
    perfs   = []
    times   = []
    for i in range(1):
        print("\t", "Step:", i)

        # Need to try-except this because sometimes bad sequences will
        # crash the simulation
        try:
            start_time = time.time()

            feature_values = all_datasets[ds_name][1].copy()
            old_fv_keys = set(feature_values.keys())

            # ReCog will often genereate 'bad' values, eg. very large ones,
            # when exp(exp(exp(...))) or a similar chain of transformations
            # is applied. This will keep sklearn from complaining.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                best_performance, root_performance = recog(
                    root_dataset = all_datasets[ds_name][0],
                    transformations = transformations,
                    number_expansions = 2000,
                    feature_values = feature_values,
                    evaluation_function = lambda x,c: score.evaluate(x,c, "decision_tree")
                )
            
            # Delete all new features for RAM
            del(feature_values)

            # Convert to percentage
            duration = time.time() - start_time
            results.append(((best_performance / root_performance) - 1) * 100)
            perfs.append(best_performance)
            times.append(duration)
        except:
            print("[ERROR] Here.")
    print()


    results_array = np.array(results)
    mean_perf = np.array(perfs).mean()
    mean = results_array.mean()
    stdd = results_array.std()
    duration = np.array(times).mean()
    all_results[ds_name] = [mean_perf, mean, stdd, duration]
    print(f" {ds_name}: {mean:>3.3f}+-{stdd:>3.3f}% improvement in {duration:>5.1f} seconds (mean).")

    with open(path, mode='w') as f:
        json.dump(all_results, f)
