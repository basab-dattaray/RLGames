import math
from collections import namedtuple

import numpy as np


def create_normalized_predictor(fn_predict_action_probablities, fn_get_valid_actions):
    def fn_get_prediction_info(state):
        action_probalities, wrapped_state_val = fn_predict_action_probablities(state)
        valid_actions = fn_get_valid_actions(state)
        if valid_actions is None:
            return action_probalities, wrapped_state_val[0], None

        action_probalities = action_probalities * valid_actions  # masking invalid moves
        sum_Ps_s = np.sum(action_probalities)
        if sum_Ps_s > 0:
            action_probalities /= sum_Ps_s  # renormalize
        else:
            action_probalities = action_probalities + valid_actions
            action_probalities /= np.sum(action_probalities)
        return action_probalities, wrapped_state_val[0], valid_actions

    return fn_get_prediction_info

def search_helper(
        state_action_qval,
        state_policy,
        state_visits
):
    EPS = 1e-8
    def fn_update_state_during_backprop(state_key, action, state_val):
        state_action_key = (state_key, action)
        if state_action_qval.fn_does_key_exist(state_action_key):  # UPDATE EXISTING
            tmp_val = (state_visits.fn_get_child_state_visits(state_action_key) * state_action_qval.fn_get_data(
                state_action_key) + state_val) / (state_visits.fn_get_child_state_visits(state_action_key) + 1)
            state_action_qval.fn_set_data(state_action_key, tmp_val)

            # Nsa[(state_action_key)] += 1
            state_visits.fn_incr_child_state_visits(state_action_key)

        else:  # UPDATE FIRST TIME
            state_action_qval.fn_set_data(state_action_key, state_val)

            # Nsa[(state_action_key)] = 1
            state_visits.fn_set_child_state_visits(state_action_key, 1)
        # Ns[state_key] += 1
        state_visits.fn_incr_state_visits(state_key)

    def fn_get_best_ucb_action(state_key, valid_moves, max_num_actions, explore_exploit_ratio):
        best_ucb = -float('inf')
        best_act = -1
        # pick the action with the highest upper confidence bound
        for action in range(max_num_actions):
            if valid_moves[action]:
                policy = state_policy.fn_get_data(state_key)
                state_action_key = (state_key, action)

                if state_action_qval.fn_does_key_exist(state_action_key):
                    # qval = cache_mgr.state_action_qval.fn_get_data(key)  # Qsa[(state_key, action)]
                    ucb = state_action_qval.fn_get_data(state_action_key) \
                          + explore_exploit_ratio * policy[action] * math.sqrt(
                        np.log(state_visits.fn_get_state_visits(state_key)) /
                              state_visits.fn_get_child_state_visits(state_action_key))

                else:
                    ucb = explore_exploit_ratio * policy[action] * math.sqrt(
                        state_visits.fn_get_state_visits(state_key) + EPS)  # Q = 0 ?
                    # u = 0
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_act = action
        action = best_act
        return action

    ret_functions = namedtuple('_', ['fn_get_best_ucb_action', 'fn_update_state_during_backprop'])
    ret_functions.fn_get_best_ucb_action = fn_get_best_ucb_action
    ret_functions.fn_update_state_during_backprop = fn_update_state_during_backprop

    return ret_functions

