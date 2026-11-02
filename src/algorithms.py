from sortedcontainers import SortedDict
from collections import defaultdict
from copy import deepcopy
import numpy as np
from sklearn.isotonic import IsotonicRegression

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

def isotonic_regression(horizon, utility, rng):
    """Online isotonic regression algorithm for learning AI confidence thresholds.

    For each unique h_t observed, fits an isotonic regressor (square loss, assumed
    non-decreasing) of y_t on b_t to estimate P(Y=1 | h_t, b_t), refit on that
    context's history at every timestep before the current observation is used
    to decide an action.
    """
    rewards = []
    actions = []
    data_b = defaultdict(list)
    data_y = defaultdict(list)

    denom = utility[1][1] - utility[0][1] + utility[0][0] - utility[1][0]
    threshold = (utility[1][1] - utility[0][1]) / denom

    for obs in horizon:
        h_t, b_t, y_t = obs

        if len(data_b[h_t]) > 0:
            model = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
            model.fit(data_b[h_t], data_y[h_t])
            p_hat = model.predict([b_t])[0]
        else:
            p_hat = 0.5

        action = int(1 - p_hat < threshold)
        actions.append(action)

        reward = utility[action][int(y_t)]
        rewards.append(reward)

        data_b[h_t].append(b_t)
        data_y[h_t].append(y_t)

    return rewards, actions

def hedge(horizon, utility, rng):
    """Online Hedge (exponential weights) algorithm for learning AI confidence thresholds.

    For each unique h_t, runs Hedge over threshold experts - one per distinct b
    value that occurs anywhere in horizon for that context - where expert bar_b
    always recommends action = int(b_t > bar_b). Since horizon is given in full
    upfront (as with all these algorithms), a single pass over it determines the
    exact, finite comparator set |B_h| before the main loop, so every expert is
    instantiated at equal weight from round 1 for its context - none is ever "new"
    mid-stream, so no discovery-order-dependent prior/handicap is needed (unlike a
    truly unbounded/growing pool, where a late-discovered but genuinely good
    threshold could be handicapped by chance alone). Weights are tracked in log
    space for numerical comfort. Every round all experts are re-weighted using the
    full-information loss (rescaled -utility) each would have incurred, and the
    action taken is a (deterministic) weighted-majority vote: each expert casts
    weight proportional to exp(log_weight) toward its recommended action, and the
    action with the larger total weight is played, with ties broken at random.
    This matters because the classical Hedge/Weighted-Majority mistake-bound
    guarantee is proven for randomized play or a majority vote over the weights -
    not for argmax/"follow-the-leader" selection, which picks only the single
    current top expert every round. Argmax has no such guarantee and can fail to
    settle when the top few experts stay close in weight (as happens under noisy/
    stochastic y_t, since even the best threshold keeps losing some rounds), even
    though the weights themselves are being updated correctly underneath.
    """
    rewards = []
    actions = []

    b_domain = defaultdict(set)
    for obs in horizon:
        h_t, b_t, y_t = obs
        b_domain[h_t].add(b_t)
    log_weights = {h: SortedDict({b: 0.0 for b in bs}) for h, bs in b_domain.items()}
    round_counts = defaultdict(int)

    u_values = [utility[a][y] for a in (0, 1) for y in (0, 1)]
    u_min, u_max = min(u_values), max(u_values)
    u_range = u_max - u_min

    for obs in horizon:
        h_t, b_t, y_t = obs
        round_counts[h_t] += 1
        t = round_counts[h_t]
        weights_h = log_weights[h_t]
        num_experts = len(weights_h)

        max_log_w = max(weights_h.values())
        w1 = 0.0
        w0 = 0.0
        for bar_b, log_w in weights_h.items():
            shifted_w = np.exp(log_w - max_log_w)
            if b_t > bar_b:
                w1 += shifted_w
            else:
                w0 += shifted_w
        if w1 > w0:
            action = 1
        elif w0 > w1:
            action = 0
        else:
            action = int(rng.integers(0, 1, endpoint=True))
        actions.append(action)

        reward = utility[action][int(y_t)]
        rewards.append(reward)

        eta = np.sqrt(np.log(num_experts) / t) if num_experts > 1 else 0.0

        for bar_b in weights_h.keys():
            expert_action = int(b_t > bar_b)
            loss = (u_max - utility[expert_action][int(y_t)]) / u_range if u_range > 0 else 0.0
            weights_h[bar_b] -= eta * loss

    return rewards, actions

def sleeping_experts_hedge(horizon, utility, rng):
    """Sleeping-experts version of the Hedge algorithm for learning AI confidence thresholds.

    Like hedge(), but the pool of threshold experts for each h_t grows online as new
    b_t values are first observed (a genuine "sleeping experts" setup: an expert is
    "asleep"/nonexistent until its threshold is first seen), rather than the full
    comparator set being known upfront. Each newly-discovered expert is seeded with a
    geometric prior weight 2^-i (i = 0-indexed discovery order for that h_t), so a
    late-discovered threshold must out-earn a handicap rather than automatically
    starting on par with established, already-tested experts. Weights are tracked in
    log space (log_weight_i(t) = -i*ln(2) - eta * cumulative_loss_i(t)) to avoid float
    underflow when many distinct b's are observed (e.g. continuous b, where it can run
    into the thousands). As in hedge(), action selection is a (deterministic)
    weighted-majority vote over currently-awake experts: each casts weight
    exp(log_weight) toward its recommended action, and the action with the larger
    total weight is played, with ties broken at random.
    """
    rewards = []
    actions = []
    log_weights = defaultdict(lambda: SortedDict())
    round_counts = defaultdict(int)
    next_rank = defaultdict(int)
    log2 = np.log(2)

    u_values = [utility[a][y] for a in (0, 1) for y in (0, 1)]
    u_min, u_max = min(u_values), max(u_values)
    u_range = u_max - u_min

    for obs in horizon:
        h_t, b_t, y_t = obs
        round_counts[h_t] += 1
        t = round_counts[h_t]
        weights_h = log_weights[h_t]

        if len(weights_h) == 0:
            action = int(rng.integers(0, 1, endpoint=True))
        else:
            max_log_w = max(weights_h.values())
            w1 = 0.0
            w0 = 0.0
            for bar_b, log_w in weights_h.items():
                shifted_w = np.exp(log_w - max_log_w)
                if b_t > bar_b:
                    w1 += shifted_w
                else:
                    w0 += shifted_w
            if w1 > w0:
                action = 1
            elif w0 > w1:
                action = 0
            else:
                action = int(rng.integers(0, 1, endpoint=True))
        actions.append(action)

        reward = utility[action][int(y_t)]
        rewards.append(reward)

        if b_t not in weights_h:
            weights_h[b_t] = -next_rank[h_t] * log2
            next_rank[h_t] += 1

        num_experts = len(weights_h)
        eta = np.sqrt(np.log(num_experts) / t) if num_experts > 1 else 0.0

        for bar_b in weights_h.keys():
            expert_action = int(b_t > bar_b)
            loss = (u_max - utility[expert_action][int(y_t)]) / u_range if u_range > 0 else 0.0
            weights_h[bar_b] -= eta * loss

    return rewards, actions