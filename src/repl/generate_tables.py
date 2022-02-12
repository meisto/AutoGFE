# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Tue 08 Feb 2022 06:42:06 PM CET
# Description: -
# ======================================================================

import numpy as np 


from src.util import score
from data import benchmark


def evaluate_model(model, dataset, evaluation_model, baselines) -> float:
    baseline = baselines[dataset, evaluation_model]

    # Transform dataset using model
    ds_transformed = dataset[0]
    ds_transformed_contex = dataset[1]

    # Evaluate transformed dataset
    performance_transformed = score.evaluate(
        ds_transformed,
        ds_transformed_contex,
        evaluation_model
    )

    # Return difference between transformed and baseline
    return performance_transformed - baseline

def get_mean_improvement_and_std(
    model,
    ds_set,
    evaluation_model,
    baselines
):
    # Lambda for readability
    ev = lambda ds: evaluation_model(model, ds, evaluation_model, baselines)

    # Evaluate model for each dataset in ds_set
    results = np.array([ev(x) for x in ds_set])

    # Return mean and stdd
    return (results.mean(), results.std())

def main():
    ## Helper functions
    latex_clean = lambda x: x.replace("_", "\\_")

    # validation_set = ["accelerometer", "ionosphere", "world_happiness"]

    # Uncomment if new baselines should be calculated
    # benchmark.generate_baselines()

    # Load baseline values
    baselines = benchmark.load_baselines()


    learning_algorithm_map = {"PPO":"PPO", "REINFORCE": "RNF"}
    evaluation_methods = ["stochastic", "greedy"]
    evaluation_model_map = {"logistic_regression": "LR", "decision_tree": "DT"}


    # Experiment A
    representation_generators   = ["quantile_data_sketch", "random_sampling"]
    rl_reward                   = ["binary_reward", "staggered_reward"]
    learning_algorithm          = ["PPO", "REINFORCE"]
    evaluation_models           = ["logistic_regression", "decision_tree"]


    
    # Stochastic evaluation
    for evaluation_method in evaluation_methods:
        latex_str = ""
        for rg in representation_generators:
            for rlr in rl_reward:
                for la in learning_algorithm:
                    for em in evaluation_models:

                        # Get model based on eval_method, rg, rlr, la
                        # model = None #TODO

                        # mean_improvement, stdd = get_mean_improvement_and_std(
                        #     model,
                        #     validation_set,
                        #     em,
                        #     baselines
                        # )
                        mean_improvement, stdd = 0.0,0.0 

                        latex_str += f"{latex_clean(rg)} & "
                        latex_str += f"{latex_clean(rlr)} & "
                        latex_str += f"{learning_algorithm_map[la]} & "
                        latex_str += f"{evaluation_model_map[em]} & "
                        latex_str += f"{mean_improvement:>.3f}$\\pm${stdd:>.3f}"
                        latex_str += "\\\\\n"

        with open(f"tables/experiment_A_{evaluation_method}.tex", mode='w') as f:
            f.write(latex_str)
            f.flush()

    # Experiment B

    # Experiment C
    datasets = sorted([
        "abalone",
        "ecoli",
        "ionosphere",
        "iris",
        "skl_boston",
        "skl_digits",
    ])

    recog = lambda _: 0.5
    latex_str = ""
    
    rf_last, rf_stdd_last = 0.0,0.0

    for ds in datasets:
        for em in evaluation_models:
            # bp = baselines[(ds,em)]
            bp, bp_stdd = baselines[(ds, em)]
            rf, rf_stdd = baselines[(ds, "random_forest")]
            ag, ag_stdd = 0.5, 0.5

            # Katch nans
            if np.isnan(bp) or np.isnan(bp_stdd):
                bp_str = "- & "
            else:
                bp_str = f"{bp:>.3f}$\\pm${bp_stdd:>.3f} & "

            # Format rf string to be empty when repeat
            if rf == rf_last and rf_stdd == rf_stdd_last:
                rf_str = "\" & "
            else:
                rf_str = f"{rf:>.3f}$\\pm${rf_stdd:>.3f} & " # random forest

            rf_last = rf
            rf_stdd_last = rf_stdd
            



            latex_str += f"{latex_clean(ds)} & " # dataset name
            latex_str += f"{evaluation_model_map[em]} &" # Evaluation model

            latex_str += bp_str # dataset baseline
            latex_str += rf_str # Random forest
            latex_str += f"{ag:>.3f}$\\pm$" # auto_gfe
            latex_str += f"{ag_stdd:>.3f}" # auto_gfe
            latex_str += "\\\\\n"

    with open("tables/experiment_C.tex", mode='w') as f:
        f.write(latex_str)
        f.flush()
    
    # rc, rc_stdd = recog(ds),0.5
    # latex_str += f"{rc:>.3f}$\\pm$" # re_cog
    # latex_str += f"{rc_stdd:>.3f} & " # re_cog
        

if __name__ == "__main__":
    main()
