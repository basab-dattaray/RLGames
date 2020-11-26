import math
from collections import namedtuple

import numpy as np


EPS = 1e-8
def search_helper(state_action_qval, state_policy,
            fn_get_state_visits,
            fn_get_child_state_visits,
            fn_does_child_state_visits_exist,
            fn_set_state_visits,
            fn_set_child_state_visits,
            fn_incr_child_state_visits
):
    def fn_get_best_ucb_action(state_key, valids, max_num_actions, explore_exploit_ratio):
        best_ucb = -float('inf')
        best_act = -1
        # pick the action with the highest upper confidence bound
        for action \
                in range(max_num_actions):
            if valids[action]:
                policy = state_policy.fn_get_data(state_key)
                state_action_key = (state_key, action)

                if state_action_qval.fn_does_key_exist(state_action_key):
                    # qval = cache_mgr.state_action_qval.fn_get_data(key)  # Qsa[(state_key, action)]
                    ucb = state_action_qval.fn_get_data(state_action_key) \
                          + explore_exploit_ratio * policy[action] * math.sqrt(
                        np.log(fn_get_state_visits(state_key)) /
                              fn_get_child_state_visits(state_action_key))

                else:
                    ucb = explore_exploit_ratio * policy[action] * math.sqrt(
                        fn_get_state_visits(state_key) + EPS)  # Q = 0 ?
                    # u = 0
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_act = action
        action = best_act
        return action

    search_helper = namedtuple('_', ['fn_get_best_ucb_action'])
    search_helper.fn_get_best_ucb_action = fn_get_best_ucb_action

    return search_helper

