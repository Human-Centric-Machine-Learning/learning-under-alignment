from collections import defaultdict
import pandas as pd
import os
import json

DEBUG = False
def find_optimal_actions(data_path, config):
    """Given the dataset and the configuration, find the optimal action for each (h, b) pair based on the expected utility."""
    data = pd.read_csv(data_path, dtype=config['data_types'])
    
    human_col = config['column_label_mapping']['human_conf_label']
    machine_col = config['column_label_mapping']['AI_conf_label']
    outcome_prob = config['column_label_mapping']['outcome_prob'] if config['use_outcome_prob'] else config['column_label_mapping']['outcome']
    if DEBUG:
        print(f"Data sample for finding optimal actions: {data.head()}")
        print(f"Human confidence column: {human_col}, Machine confidence column: {machine_col}, Outcome probability column: {outcome_prob}")

    optimal_actions_dict = {}
    if config['use_outcome_prob']:
        p_0_given_h_b = 1 - (data[[human_col, machine_col, outcome_prob]].groupby([human_col, machine_col]).mean() / 100)
        p_1_given_h_b = 1 - p_0_given_h_b
        mu_0_given_h_b = p_0_given_h_b * config['utility'][0][0] + p_1_given_h_b * config['utility'][0][1]
    else:
         # add one column with the utility function value for each row given action = 0
        data['utility_a_0'] = data.apply(lambda row: config['utility'][0][int(row[config['column_label_mapping']['outcome']])], axis=1)        
        mu_0_given_h_b = data[[human_col, machine_col, 'utility_a_0']].groupby([human_col, machine_col]).mean()
    
    if DEBUG:
        print(f"mu_0_given_h_b sample: {mu_0_given_h_b.head()}")
    
    c = (config['utility'][1][1] - config['utility'][0][1]) / (config['utility'][1][1] - config['utility'][1][0] + config['utility'][0][0] - config['utility'][0][1])
    threshold =  c * config['utility'][0][0]  + (1 - c) * config['utility'][0][1] 
    
        
    if DEBUG:
        print(f"Threshold of expected utility for optimal actions: {threshold}")
    
    if 'utility_a_0' not in mu_0_given_h_b.columns:
        optimal_actions_dict_raw = (mu_0_given_h_b <= threshold).astype(int).to_dict()[config['column_label_mapping']['outcome_prob']]
        if DEBUG:
            print(f"Optimal actions for each (h, b) pair: {optimal_actions_dict_raw}")
        optimal_actions_dict = optimal_actions_dict_raw
    else:
        optimal_actions_dict_raw = (mu_0_given_h_b <= threshold).astype(int).to_dict()['utility_a_0']
        optimal_actions_dict = defaultdict(lambda:{})
        if DEBUG:
            print(f"Optimal actions for each (h, b) pair: {optimal_actions_dict_raw}")
    optimal_actions_dict = defaultdict(lambda:{})
        
    for (h, b), action in optimal_actions_dict_raw.items():
        optimal_actions_dict[h][b] = action

    # change the of optimal_actions_dict to be standard dict
    # also convert the keys to string for json serialization
    optimal_actions_dict = {str(h): {str(b): action for b, action in b_dict.items()} for h, b_dict in optimal_actions_dict.items()} 
    if DEBUG:
        print(f"Optimal actions for each (h, b) pair: {optimal_actions_dict}")
    return optimal_actions_dict

def get_optimal_actions(data_path, config):
    """Get the optimal actions for each (h, b) pair, either by loading from an existing file or by computing it from the dataset."""
    def _convert(d):
        out = {}
        for k, v in d.items():
            k2 = str(k)
            if isinstance(v, dict):
                out[k2] = _convert(v)
            else:
                try:
                    out[k2] = int(v)
                except (TypeError, ValueError):
                    out[k2] = v
        return out

    if os.path.exists(f"{config['optimal_path_actions']}/{data_path.split('/')[-1].split('.')[0]}.json"):
        with open(f"{config['optimal_path_actions']}/{data_path.split('/')[-1].split('.')[0]}.json", 'r') as f:
            # optimal_actions_dict = json.load(f)
            optimal_actions_dict = json.load(f, object_hook=_convert)
    else:
        optimal_actions_dict = find_optimal_actions(data_path, config)
        os.makedirs(config['optimal_path_actions'], exist_ok=True)
        with open(f"{config['optimal_path_actions']}/{data_path.split('/')[-1].split('.')[0]}.json", 'w') as f:
            json.dump(optimal_actions_dict, f, indent=4)
    return optimal_actions_dict