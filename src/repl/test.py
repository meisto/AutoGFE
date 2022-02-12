# Stdlib imports
import json
import os
from typing import List, Tuple

# library imports
import numpy as np
import pandas as pd

# local imports
from src.agent.rollout import evaluate, transform_dataset
from src.util.dataset_loader import load_datasets
from src.util.experiment import Experiment, load_savepoint
from src.util.score import evaluate as ev_function
from src.transformations import real

# Load metadata
PARAMETER_PATH = os.getenv("PARAMETER_PATH")
EXPERIMENT_PATH = os.getenv("EXPERIMENT_PATH")
VALIDATION_DATA_PATH = os.getenv("VALIDATION_DATA_PATH")

# Set defaults
if PARAMETER_PATH == None: PARAMETER_PATH = "/home/tobias/ma/parameters"
if EXPERIMENT_PATH == None: EXPERIMENT_PATH = "/home/tobias/ma/experiments"
if VALIDATION_DATA_PATH == None: VALIDATION_DATA_PATH = "/home/tobias/ma/data/validation"

# local imports

model_dir = os.path.join(EXPERIMENT_PATH, "params_007/")
model_file_name = "-7286741115737232906step_0325.sd"

# Get the experiment file
files = next(os.walk(model_dir))[2]
files = [x for x in files if x.endswith(".exp")]
assert len(files) == 1, "To many experiment files"

# Load experiment with good parameters
exp = load_savepoint(model_dir)
exp.load_agent_parameters(os.path.join(model_dir, model_file_name))


validation_data = load_datasets(VALIDATION_DATA_PATH)
evaluation_function = lambda x,c: ev_function(dataset=x, context=c, modeltype="decision_tree")
tfds = lambda ds, fv: transform_dataset(
    root_dataset = ds,
    feature_values = fv,
    evaluation_function = evaluation_function,
    experiment = exp,
    predict = lambda agent, repre: agent(repre)
)

for vdata, context in validation_data:
    res = [tfds(vdata, context) for _ in range(10)]
    res = np.array(res)
    
    print(
        "| ", 
        vdata.name.rjust(20), " | ",
        f"{evaluation_function(vdata, context):1.3f} | ",
        # f"{evaluation_function(tfds(vdata, context), context):1.3f}"
        f"{res.mean():1.3f}+-{res.std():1.3f}"
    )
    # tfds(vdata, context)
