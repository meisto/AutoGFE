# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 12 Jan 2022 10:07:55 PM CET
# Description: -
# ======================================================================
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


import src.util.datasets as D
import src.plot as P
from src.agent.autogfe_transformer import AutoGFETransformer
from src.env.feature import Dataset, Feature
import src.util.experiment as E
from src.util.experiment import load_experiment, Experiment
from src.util.score import evaluate
from main import load_basics

def get_base_performance(
    dataset: Dataset,
    context: Dict[Feature, np.ndarray],
) -> float:
    
    # a = evaluate(dataset, context, "logistic_regression")
    b = evaluate(dataset, context, "decision_tree")
    
    #TODO: change this
    return b

def get_random_forest_performance(
    dataset: Dataset,
    context: Dict[Feature, np.ndarray],
) -> float:
    return evaluate(dataset, context, "random_forest")


def get_best_performance_table(
    datasets: List[Tuple[Dataset,pd.DataFrame]],
    trans: AutoGFETransformer,
):
    dss = sorted(datasets, key=lambda x: x[0].name)

    print("Start generating AutoGFE performance values...")
    performance = []
    for index in range(len(dss)):
        ds, context = dss[index]
        i = "|" + ("+" * index) + (" " * (len(dss) - index)) + "| " + ds.name
        print(i)

        p = trans.get_performance(ds, context)
        performance.append(p)

    print()

    # Generate comparison values
    print("Start generating comparison values...")
    base_performances   = [get_base_performance(ds[0], ds[1]) for ds in dss]
    random_forest       = [get_random_forest_performance(ds[0], ds[1]) for ds in dss]


    df_dss = pd.DataFrame({
        "dataset"           : [x[0].name for x in dss],
        "base_performance"  : base_performances,
        "mlp"               : [0.0 for _ in dss],
        "random_forest"     : random_forest,
        "performance"       : performance,
    })

    df_dss["relative_gain"] = df_dss["performance"] / df_dss["base_performance"]
    df_dss["relative_gain"] = (df_dss["relative_gain"] - 1)*100

    res_str = ""
    for row in df_dss.itertuples():
        ds_name = row[1].replace("_", "\\_")
        rel_performance = f"{'+' if row[6] > 0 else ''}{row[6]:>.2f}\\%"
        
        # Assemble table entry lines
        res_str += "      "
        res_str += f"{ds_name} & "
        res_str += f"{row[2]:>.3f} & "  # Without FE
        res_str += f"{row[3]:>.3f} & "  # MLP
        res_str += f"{row[4]:>.3f} & "  # Random Forest
        res_str += f"{row[5]:>.3f} & "  # Our performance 
        res_str += f"{rel_performance}\\\\\n"

    template = open("../text/tables/dataset_performance_raw.tex", mode='r').read()

    with open("../text/tables/dataset_performance.tex", mode='w') as f:
        res_str = template.replace("%%ENTRY%%", res_str)
        f.write(res_str)
        f.flush()

def get_mode_table():
    # Load overview dataframe
    exp_df = P.get_experiment_df()

    # Load all experiments
    exps = [E.load_experiment(e) for e in exp_df.root_path.unique()]

    # Print table containing results for given hyperparameters
    param_df = pd.DataFrame({
        "representation": [e.parameters["representation_generator"] for e in exps],
        "reward_type": [e.parameters["reward_function"] for e in exps],
        "representation": [e.parameters["representation_generator"] for e in exps],
        "name": [e.name for e in exps],
        "root_paths": [e.root_path for e in exps],
    })

    train = []
    test = []

    for _, x in param_df[["name", "root_paths"]].iterrows():
        name = x["name"]
        path = x["root_paths"]

        pdf = P.get_performance_df(root_path=path, interactive=False)
        pdf = pdf[pdf["test_performances"] == pdf["test_performances"].max()].iloc[0]
        
        train.append(pdf["train_performances"].item())
        test.append(pdf["test_performances"].item())

    param_df["train"] = train
    param_df["test"] = test

    param_df.drop(columns=["name","root_paths"], inplace=True)


    df = param_df
    df.reset_index(inplace=True)

    # Generate a latex string
    res_str = ""
    rer = sorted(param_df["representation"].unique())
    rew = sorted(param_df["reward_type"].unique())
    for r1 in rer:
        for r2 in rew:
            # Select all fitting rows
            selector = (df["representation"] == r1) & (df["reward_type"] == r2)
            max_element = df[selector]["test"].max()

            # Get the most fitting row
            u = df[selector & (df["test"] == max_element)].iloc[0]

            # Generate latex string line
            representation = u["representation"]
            reward = u["reward_type"]

            train   = f"{u['train']:>.3f}"
            test    = f"{u['test']:>.3f}"
            veri    = f"{0:>.3f}"
            boost   = 0.0
            boost = f"+{boost:>.3f}%" if boost >= 0 else f"{boost:>.3f}%"
        
            res_str += f"{representation} & {reward} & mixed & {train} & "
            res_str += f"{test} & {veri} & {boost}\\\\\n"

    # Clean string up
    res_str = res_str.replace("_", "\\_")
    res_str = res_str.replace("%","\\%")

    # Load template
    template = open("../text/tables/mode_performance_raw.tex", mode='r').read()

    # Generate file
    with open("../text/tables/mode_performance.tex", mode='w') as f:
        res_str = template.replace("%%ENTRY%%", res_str)
        f.write(res_str)
        f.flush()



if __name__ == "__main__":
    get_mode_table()
    import sys
    sys.exit()

    # Load datasets
    print("Start loading all datasets...")
    (train, test), _ = load_basics()
    train.extend(test)
    dss = train

    print("Start loading best performing model...")
    exp_df = P.get_experiment_df()
    exp_df["res"] = exp_df[["latest_test_performance", "latest_train_performance"]].mean(axis=1)
    max_performer = exp_df[exp_df["res"] == exp_df["res"].max()]
    assert max_performer.shape[0] == 1, "Too many max performers."

    root_path = max_performer["root_path"].item()
    experiment: Experiment = load_experiment(root_path)

    policy = experiment.load_savepoint()

    print("Start generating a transformer...")
    transformer = AutoGFETransformer(policy=policy, number_paths=10, max_steps=10, exp=experiment)

    print("Start generating the table...")
    get_best_performance_table(dss, transformer)

    # print("Start generating mode table...")
    # get_mode_table()
