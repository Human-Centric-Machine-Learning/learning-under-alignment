import os
from collections import defaultdict
from utils import get_optimal_actions
from algorithms import algorithm_one, vanilla_contextual, optimal_action, isotonic_regression, follow_the_leader
from copy import deepcopy
import json 
import numpy as np
import pandas as pd 
import argparse
import tqdm 

DEBUG = False

def single_experiment(algorithm_name, dataset, rng, num_iterations, utility, optimal_action_dict={}, use_outcome_prob=True, Bmin=1, Bmax=100, b_int=True, horizon_seed=(0,)):
    """Run a single algorithm on a single dataset for num_iterations
       iterations and return the rewards and actions.

       Arguments:
       algorithm_name: str, name of the algorithm to run. One of 
                       "algorithm_one", "vanilla_contextual", "optimal"
       dataset: List, list of tuples (h_t, b_t, p_y) or (h_t, b_t, y_t) representing the horizon
       rng: np.random.Generator, random number generator for any randomness in the algorithm
       num_iterations: int, number of iterations to run the algorithm for
       utility: List[List], 2x2 list representing the utility matrix
       optimal_action_dict: Dict, dictionary mapping human confidence levels and machine confidence values to optimal actions
       use_outcome_prob: bool, whether to use the outcome probabilities in the dataset (if True, the dataset is assumed to have p_y in the third column; if False, the dataset is assumed to have y in the outcome_prob column)
       Bmin: minimum machine confidence  (only used for algorithm_one)
       Bmax: maximum machine confidence  (only used for algorithm_one)
       b_int: whether to use integer machine confidence values (only used for algorithm_one)
       horizon_seed: Tuple, seed for the shuffle and sampled outcomes of each iteration; it does not
                     depend on the algorithm, so all algorithms (and the optimal policy) see the same
                     horizon in a given iteration and the regret is computed on identical data

       Returns:
       results: Dict, dictionary containing the rewards and actions for each timestep in each iteration
    """
    b_domain = None
    if algorithm_name == "follow_the_leader":
        # the comparator set of threshold experts per context h_t is fixed by
        # the dataset itself, not by iteration order, so read it once here
        # instead of re-deriving it from the shuffled horizon every iteration
        b_domain = defaultdict(set)
        for h_t, b_t, _ in dataset:
            b_domain[h_t].add(b_t)

    results = []
    for iteration in tqdm.tqdm(range(num_iterations)):
        if DEBUG:
            print(f"Running iteration {iteration} for algorithm {algorithm_name}")

        # shuffle the dataset (same shuffle and outcomes for every algorithm in this iteration)
        horizon_rng = np.random.default_rng([*horizon_seed, iteration])
        shuffled_dataset = horizon_rng.permutation(deepcopy(dataset))

        if use_outcome_prob:
            # samples outcomes based on the probabilities in the dataset
            probabilities = shuffled_dataset[:, 2].astype(float) / 100 # assuming the probabilities are in the third column and are between 0 and 100
            if DEBUG:
                print(f"Probability sample at iteration {iteration}: {probabilities[:5]}")
            outcomes = horizon_rng.binomial(1, probabilities)
            # create the horizon as a list of tuples (h_t, b_t, y_t)
            shuffled_dataset[:, 2] = outcomes
        horizon = shuffled_dataset

        if DEBUG:
            print(f"Horizon sample at iteration {iteration}: {horizon[:5]}")

        if algorithm_name == "algorithm_one":
            rewards, actions = algorithm_one(horizon, utility, rng, Bmin=Bmin, Bmax=Bmax, b_int=b_int)
        elif algorithm_name == "vanilla_contextual":
            rewards, actions = vanilla_contextual(horizon, utility, rng)
        elif algorithm_name == "isotonic_regression":
            rewards, actions = isotonic_regression(horizon, utility, rng)
        elif algorithm_name == "follow_the_leader":
            rewards, actions = follow_the_leader(horizon, utility, rng, b_domain=b_domain)
        elif algorithm_name == "optimal":
            rewards, actions = optimal_action(horizon, utility, optimal_action_dict)
        else:
            raise ValueError(f"Invalid algorithm name: {algorithm_name}")
        
        if DEBUG:
            print(f"Reward sample at iteration {iteration} for algorithm {algorithm_name}: {rewards[:5]}")
            print(f"Action sample at iteration {iteration} for algorithm {algorithm_name}: {actions[:5]}")
        t = 1
        for r, a in zip(rewards, actions):
            if algorithm_name != "algorithm_one":
                results.append({
                    "iteration": iteration,
                    "timestep": t,
                    "reward": r,
                    "action": a,
                    "algorithm": algorithm_name
                })
            else:
                # for algorithm_one, we log the b_threshold 
                # instead of the action taken
                results.append({
                    "iteration": iteration,
                    "timestep": t,
                    "reward": r,
                    "b_threshold": a,
                    "algorithm": algorithm_name
                })
            t += 1

    if DEBUG:        
        print(f"Results sample for algorithm {algorithm_name}: {results[:5]}")
    return results


def run(config_file="configs/human_alignment.json", job_id=0):
    """Run the experiment for one algorithm and dataset and save the results to a json file.
    
       Arguments:
       config_file: str, path to the experiment configuration file
       job_id: int, id of the job to run (used to select the algorithm and dataset)
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # configure logging
    if DEBUG:
        os.makedirs(config["log_dir"], exist_ok=True)
        log_file = f"{config['log_dir']}/job_{job_id}.log"
        print(f"Logging output to {log_file}")
        # set the std output to a log file
        log_f = open(log_file, 'w')
        os.dup2(log_f.fileno(), 1)
    
    all_jobs = len(config["algorithms"]) * len(config["datasets"])
    if job_id >= all_jobs:
        # reverse the std output to the terminal
        os.dup2(1, 2)
        raise ValueError(f"Invalid job_id: {job_id}. Must be between 0 and {all_jobs - 1}")

    # create the random generator for this job  
    master_rng = np.random.default_rng(seed=config["master_seed"])
    all_seeds = master_rng.integers(1e7, 1e8, size=all_jobs)
    rng = np.random.default_rng(seed=all_seeds[job_id])

    # select the algorithm and dataset for this job
    algorithm_name = config["algorithms"][job_id // len(config["datasets"])]
    dataset_file = config["datasets"][job_id % len(config["datasets"])]
    if DEBUG:
        print(f"Running job {job_id} with algorithm {algorithm_name} on dataset {dataset_file}")

    # load the dataset
    dataset_path = f"{config['data_dir']}/{dataset_file}"
    dataset = pd.read_csv(dataset_path, dtype=config['data_types'])[list(config['column_label_mapping'].values())].values
    if DEBUG:
        print(f"Dataset sample for job {job_id}: {dataset[:5]}")
    
    optimal_action_dict = {}
    if algorithm_name == "optimal":
        # get the optimal action dictionary for the optimal algorithm
        optimal_action_dict = get_optimal_actions(dataset_path, config)

        if DEBUG:
            print(f"optimal_action_dict for job {job_id}: {optimal_action_dict}")
    
    results = single_experiment(
                                algorithm_name, 
                                dataset, 
                                rng, 
                                config["num_iterations"], 
                                config["utility"], 
                                optimal_action_dict=optimal_action_dict,
                                use_outcome_prob=config["use_outcome_prob"],
                                Bmin=config["Bmin"],
                                Bmax=config["Bmax"],
                                b_int=config["b_int"],
                                horizon_seed=(config["master_seed"], job_id % len(config["datasets"]))
                            )

    # save results to json file
    output_path = f"{config["output_dir"]}/{config['num_iterations']}iters/{dataset_file.split('.')[0]}/{algorithm_name}"
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/results.json", 'w') as f:
        json.dump(results, f, indent=4)

    if DEBUG:    
        # reverse the std output to the terminal
        os.dup2(1, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/human_alignment.json", help="Path to the experiment configuration file")
    parser.add_argument("--job_id", type=int, default=0, help="ID of the job to run (used to select the algorithm and dataset)")
    args = parser.parse_args()
    run(config_file=args.config_file, job_id=args.job_id)
    