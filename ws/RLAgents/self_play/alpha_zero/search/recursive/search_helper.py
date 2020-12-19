import math
from collections import namedtuple
import numpy as np

from ws.RLAgents.self_play.alpha_zero.search.cache_mgt import cache_mgt
from ws.RLAgents.self_play.alpha_zero.search.recursive.state_visit_mgt import state_visit_mgt


# def create_normalized_predictor(fn_predict_policies, fn_get_valid_actions):


def search_helper(
        args,
        game_mgr,
        neural_net_mgr,
):
    EPS = 1e-8
    fn_terminal_value = lambda pieces: game_mgr.fn_get_game_progress_status(pieces, 1)
    fn_get_valid_actions = lambda board: game_mgr.fn_get_valid_moves(board, 1)

    cache_mgr = cache_mgt()

    state_visits = state_visit_mgt()

    def fn_get_real_state_value(state):
        state_key = game_mgr.fn_get_state_key(state)
        if not cache_mgr.state_results.fn_does_key_exist(state_key):
            cache_mgr.state_results.fn_set_data(state_key, fn_terminal_value(state))
        return cache_mgr.state_results.fn_get_data(state_key)

    def fn_get_predicted_based_state_value(state):
        state_key = game_mgr.fn_get_state_key(state)
        is_new_prediction = False
        if not cache_mgr.state_info.fn_does_key_exist(state_key):
            is_new_prediction = True
            # leaf node
            policy, state_val, valid_actions = fn_get_prediction_info_3(state)
            if valid_actions is None:
                return -state_val
            state_info = {
                'policy': policy,
                'state_val': state_val,
                'valid_actions': valid_actions
            }
            cache_mgr.state_info.fn_set_data(state_key, state_info)
            cache_mgr.state_policy.fn_set_data(state_key, policy)

            cache_mgr.state_valid_moves.fn_set_data(state_key, valid_actions)

            # Ns[state_key] = 0
            state_visits.fn_set_state_visits(state_key, 0)

        state_info = cache_mgr.state_info.fn_get_data(state_key)

        return state_info, is_new_prediction

    def fn_get_prediction_info_3(state):
        action_probalities, wrapped_state_val = neural_net_mgr.fn_neural_predict(state)
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

    def fn_update_state_during_backprop(state_key, action, state_val):
        state_action_key = (state_key, action)
        if cache_mgr.state_action_qval.fn_does_key_exist(state_action_key):  # UPDATE EXISTING
            tmp_val = (state_visits.fn_get_child_state_visits(state_action_key) * cache_mgr.state_action_qval.fn_get_data(
                state_action_key) + state_val) / (state_visits.fn_get_child_state_visits(state_action_key) + 1)
            cache_mgr.state_action_qval.fn_set_data(state_action_key, tmp_val)

            # Nsa[(state_action_key)] += 1
            state_visits.fn_incr_child_state_visits(state_action_key)

        else:  # UPDATE FIRST TIME
            cache_mgr.state_action_qval.fn_set_data(state_action_key, state_val)

            # Nsa[(state_action_key)] = 1
            state_visits.fn_set_child_state_visits(state_action_key, 1)
        # Ns[state_key] += 1
        state_visits.fn_incr_state_visits(state_key)

    def fn_get_best_ucb_action(state_key, max_num_actions, explore_exploit_ratio):
        valid_moves = cache_mgr.state_valid_moves.fn_get_data(state_key)

        best_ucb = -float('inf')
        best_act = -1

        action_prob_for_exploration = 1

        # pick the action with the highest upper confidence bound
        for action in range(max_num_actions):

            if valid_moves[action]:
                policy = cache_mgr.state_policy.fn_get_data(state_key)
                state_action_key = (state_key, action)

                if args.mcts_ucb_use_action_prob_for_exploration:
                    action_prob_for_exploration = policy[action]

                if cache_mgr.state_action_qval.fn_does_key_exist(state_action_key):
                    parent_visit_factor = state_visits.fn_get_state_visits(state_key)
                    if args.mcts_ucb_use_log_in_numerator:
                        parent_visit_factor = np.log(parent_visit_factor)

                    ucb = cache_mgr.state_action_qval.fn_get_data(state_action_key) \
                          + explore_exploit_ratio * action_prob_for_exploration * math.sqrt\
                                (
                                    parent_visit_factor / state_visits.fn_get_child_state_visits(state_action_key)
                                )

                else:
                    ucb = explore_exploit_ratio * action_prob_for_exploration * math.sqrt(
                        state_visits.fn_get_state_visits(state_key) + EPS)  # Q = 0 ?
                    # u = 0
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_act = action
        action = best_act
        return action

    ret_functions = namedtuple('_', [
        'cache_mgr',
        'state_visits',
        'fn_get_best_ucb_action',
        'fn_update_state_during_backprop',
        # 'fn_get_prediction_info_3',

        'fn_get_real_state_value',
        'fn_get_predicted_based_state_value',
    ])
    ret_functions.cache_mgr = cache_mgr
    ret_functions.state_visits = state_visits
    ret_functions.fn_get_best_ucb_action = fn_get_best_ucb_action
    ret_functions.fn_update_state_during_backprop = fn_update_state_during_backprop
    # ret_functions.fn_get_prediction_info_3 = fn_get_prediction_info_3

    ret_functions.fn_get_real_state_value = fn_get_real_state_value
    ret_functions.fn_get_predicted_based_state_value = fn_get_predicted_based_state_value

    return ret_functions

