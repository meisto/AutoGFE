# AutoGFE
Repository for the Masters Thesis AutoGFE - Automatic Feature Engineering

**If you are an examiner of this Thesis:**
The last commit on the termination date has the hash: 'b8cca91d12c5d549b27b3214db0d6fa6cf1356aa'. I might make some later commits to clean up the code a bit and to add missing source links. 


## Code
This folder contains the code that belongs to this master thesis and, as such, is quite extensive. Therefore this markdown file is meant to be a general outline of what is where, how it is structured and how you can run it.

## General

## Structure
The code in this project is structured in the following way: the folder this document is in is the root folder. It contains the *README.md* and *main.py* files. It also contains the *src* and *data* folders.
- The **source** folder contains most of the code.
   - **src/agent**:
   - **src/env**:
   - **src/repl**:
   - **src/representation**:
   - **src/reproductions**:
   - **src/transformations**:
   - **src/util**:
- The **data** folder contains the datasets as .csv files. It also contains a list of datasets. 
   - ionosphere
```bash
.
├── data
├── data_new
│   └── dataset_list.md
├── experiments
│   └── params_004
│       ├── -4437039769131941435.exp
│       ├── -4437039769131941435.sd
│       ├── -4437039769131941435step_0000.sd
│       └── -4437039769131941435step_0004.sd
├── main.py
├── parameters
│   ├── generate_parameters.py
│   └── params_004.json
├── README.md
├── src
│   ├── agent
│   │   ├── generate_policy.py
│   │   ├── gin_layer.py
│   │   ├── PPO_agent.py
│   │   ├── PPO.py
│   │   ├── random_tree.py
│   │   ├── REINFORCE_agent.py
│   │   ├── REINFORCE.py
│   │   └── rollout.py
│   ├── env
│   │   ├── feature.py
│   │   ├── reward.py
│   │   ├── ttenv.py
│   │   └── ttree.py
│   ├── repl
│   │   ├── baselines.py
│   │   ├── generate_tables.py
│   │   └── visualizer.ipynb
│   ├── representation
│   │   ├── binning.py
│   │   ├── sampling.py
│   │   └── v2.py
│   ├── reproductions
│   │   └── recog.py
│   ├── transformations
│   │   ├── categorical.py
│   │   ├── date.py
│   │   ├── real.py
│   │   └── transformation.py
│   └── util
│       ├── cond.py
│       ├── experiment.py
│       ├── importer.py
│       ├── logger.py
│       ├── rl_util.py
│       ├── score.py
│       └── tmp.py
└── tables

33 directories, 84 files
```
## How to run
- Load or set up a python venv compatible with python 3.8.10.
- Install dependencies in **requirements.txt**
- Run the following command in bash.
```bash
export EXPERIMENT_PATH="/tmp/experiments/" # Change this path according to os
mkdir $EXPERIMENT_PATH
# chmod +x main.py # Modify access right if needed.
exec main.py

```
*Note:* In some instances, the implementation will turn unstable (eg. when all features are randomly selected for feature selection.)

## Dependencies, build and scripts
This is a python project for python3. It has been tested and developed on a Ubuntu 20.04.02 LTS machine using python3 3.8.10. 

There might be some shell scripts included. These were developed for zsh but should be mostly POSIX compliant. 
