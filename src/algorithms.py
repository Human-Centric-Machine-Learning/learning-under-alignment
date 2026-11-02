from sortedcontainers import SortedDict
from collections import defaultdict
from copy import deepcopy
import numpy as np

DEBUG = False

def algorithm_one(horizon, utility, rng, Bmin=1, Bmax=100, b_int=True):
    """Our algorithm for learning AI confidence thresholds.
    
    Arguments:
        horizon: List, list of tuples (h_t, b_t, y_t) representing the horizon
        utility: List[List], 2x2 list representing the utility matrix
        rng: np.random.Generator, random number generator for any randomness in the algorithm
        Bmin: minimum machine confidence  
        Bmax: maximum machine confidence  
        b_int: whether to use integer machine confidence thresholds 

    Returns:
        rewards: List, list of rewards received at each timestep
        b_bar: List, list of chosen machine confidence thresholds at each timestep

    """
    rewards = []
    b_bar = []
    estimators  = defaultdict(lambda: SortedDict())
    for t, obs in enumerate(horizon):
        h_t, b_t, y_t = obs
        
        # get the b bar with the highest estimator for h_t
        if h_t in estimators:
            max_estimator_value_indices = np.argmax(estimators[h_t].values(), keepdims=True)
            if DEBUG:
                print(f"Max estimator value indices for h_t={h_t} at timestep {t}: {max_estimator_value_indices}")
            # pick randomly if there are multiple intervals with the same estimator value
            b_bar_idx = rng.choice(max_estimator_value_indices)
            b_bar_low = estimators[h_t].peekitem(b_bar_idx)[0]
            b_bar_high = Bmax if b_bar_idx == len(estimators[h_t]) - 1 else estimators[h_t].peekitem(b_bar_idx + 1)[0]
            # select a random b bar in the interval
            if b_int:
                b_bar_t = rng.integers(b_bar_low, b_bar_high, endpoint=False, dtype=int)
            else:
                b_bar_t = rng.uniform(b_bar_low, b_bar_high)
        else:
            if b_int:
                b_bar_t = rng.integers(Bmin, Bmax + 1, endpoint=True, dtype=int)
            else:
                b_bar_t = rng.uniform(Bmin, Bmax)
        
        b_bar.append(b_bar_t)

        reward = utility[int(b_t > b_bar_t)][int(y_t)]
        rewards.append(reward)

        if DEBUG:
            print(f"Observation at timestep {t}: h_t={h_t}, b_t={b_t}, y_t={y_t}")
            print(f"Chosen b_bar at timestep {t}: {b_bar_t}")
            print(f"Reward received at timestep {t}: {reward}")
            print(f"Estimators before update at timestep {t} for h_t={h_t}: {estimators[h_t]}")
        if b_t not in estimators[h_t]:
            # initialize the estimator values
            if len(estimators[h_t]) == 0:
                estimators[h_t][0] = 0
            if t == 0:
                estimators[h_t][b_t] = 0
            else:
                b_t_index = estimators[h_t].bisect_right(b_t)
                if DEBUG:
                    print(f"b_t_index for b_t={b_t} at timestep {t}: {b_t_index}")
                # assumes that b_t > 0
                b_prev = estimators[h_t].peekitem(b_t_index - 1)[0]
                estimators[h_t][b_t] = deepcopy(estimators[h_t][b_prev])
        for b in estimators[h_t].keys():
            estimators[h_t][b] += utility[int(b < b_t)][int(y_t)]
        if DEBUG:            
            print(f"Estimators after update at timestep {t} for h_t={h_t}: {estimators[h_t]}")
    return rewards, b_bar

def vanilla_contextual(horizon, utility, rng):
    """A vanilla contextual learning algorithm that chooses the action with the highest average reward for each (h, b) pair."""
    rewards = []
    estimators  = defaultdict(lambda: [0, 0])
    actions = []
    for obs in horizon:
        h_t, b_t, y_t = obs

        if not any(estimators[(h_t, b_t)]):
            action = rng.integers(0, 1, endpoint=True, dtype=int)
        else:
            action = int(estimators[(h_t, b_t)][0] < estimators[(h_t, b_t)][1])
        actions.append(action)
        
        reward = utility[action][int(y_t)]
        rewards.append(reward)

        estimators[(h_t, b_t)][action] += reward
        estimators[(h_t, b_t)][1 - action] += utility[1 - action][int(y_t)]

    
    return rewards, actions


def optimal_action(horizon, utility,  action_policy_dict):
    """Optimal decision policy"""
    rewards = []
    actions = []
    for obs in horizon:
        h_t, b_t, y_t = obs
        action = action_policy_dict[str(h_t)][str(b_t)]
        rewards.append(utility[action][int(y_t)])
        actions.append(action)
    return rewards, actions