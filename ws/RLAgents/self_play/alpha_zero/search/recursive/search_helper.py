import math
from collections import namedtuple

import numpy as np

from ws.RLAgents.self_play.alpha_zero.search.cache_mgt import cache_mgt
from ws.RLAgents.self_play.alpha_zero.search.recursive.state_visit_mgt import state_visit_mgt


def search_helper(
        args,
        game_mgr,
        neural_net_mgr,
):
    EPS = 1e-8

    cache_mgr = cache_mgt()

    state_visits = state_visit_mgt()

    def fn_get_visit_counts(state_key):
        counts = [state_visits.fn_get_child_state_visits((state_key, a))
                  if state_visits.fn_does_child_state_visits_exist((state_key, a)) else 0 for a in
                  range(game_mgr.fn_get_action_size())]
        return counts

    def fn_get_cached_allowed_moves(state):
        fn_get_allowed_moves = lambda s: game_mgr.fn_get_valid_moves(s, player=1)

        state_key = game_mgr.fn_get_state_key(state)
        if not cache_mgr.s_allowed_moves.fn_does_key_exist(state_key):
            allowed_moves = fn_get_allowed_moves(state)
            cache_mgr.s_allowed_moves.fn_set_data(state_key, allowed_moves)
            return allowed_moves
        else:
            return cache_mgr.s_allowed_moves.fn_get_data(state_key)

    def fn_get_cached_results(state):
        fn_get_progress_status = lambda s: game_mgr.fn_get_game_progress_status(s, player=1)

        state_key = game_mgr.fn_get_state_key(state)
        if not cache_mgr.s_results.fn_does_key_exist(state_key):
            cache_mgr.s_results.fn_set_data(state_key, fn_get_progress_status(state))
        return cache_mgr.s_results.fn_get_data(state_key)

    def fn_visit_new_state_if_possible(state):
        state_key = game_mgr.fn_get_state_key(state)
        if not cache_mgr.s_predictions.fn_does_key_exist(state_key):
            # leaf node
            policy, state_val, moves_are_allowed = fn_get_cached_predictions(state)
            if not moves_are_allowed:
                return - state_val

            s_predictions = {
                'policy': policy,
                'state_val': state_val,
            }
            cache_mgr.s_predictions.fn_set_data(state_key, s_predictions)

            state_visits.fn_set_state_visits(state_key, 0)

            return s_predictions

        return None

    def fn_get_cached_predictions(state):
        action_probalities, wrapped_state_val = neural_net_mgr.fn_neural_predict(state)
        allowed_moves = fn_get_cached_allowed_moves(state)
        moves_are_allowed = True
        if allowed_moves is None:
            moves_are_allowed = False
            return action_probalities, wrapped_state_val[0], moves_are_allowed

        action_probalities = action_probalities * allowed_moves  # masking disallowed moves
        sum_action_probabilities = np.sum(action_probalities)
        if sum_action_probabilities > 0:
            action_probalities /= sum_action_probabilities  # re-normalize
        else:
            action_probalities = action_probalities + allowed_moves
            action_probalities /= np.sum(action_probalities)
        return action_probalities, wrapped_state_val[0], moves_are_allowed

    def fn_update_state_during_backprop(state_key, action, state_val):
        state_action_key = (state_key, action)
        if not cache_mgr.sa_qval.fn_does_key_exist(state_action_key):  # CREATE NEW STATE-ACTION
            cache_mgr.sa_qval.fn_set_data(state_action_key, state_val)
        else:
            tmp_val = (state_visits.fn_get_child_state_visits(
                state_action_key) * cache_mgr.sa_qval.fn_get_data(
                state_action_key) + state_val) / (state_visits.fn_get_child_state_visits(state_action_key) + 1)
            cache_mgr.sa_qval.fn_set_data(state_action_key, tmp_val)

        state_visits.fn_incr_child_state_visits(state_action_key)
        state_visits.fn_incr_state_visits(state_key)

    def fn_get_best_ucb_action(state_key):
        allowed_moves = cache_mgr.s_allowed_moves.fn_get_data(state_key)

        best_ucb = -float('inf')
        best_act = None

        action_prob_for_exploration = 1

        # pick the action with the highest upper confidence bound
        for action in range(game_mgr.fn_get_action_size()):

            if allowed_moves[action] != 0:
                s_predictions = cache_mgr.s_predictions.fn_get_data(state_key)

                policy = s_predictions['state_val']
                state_action_key = (state_key, action)

                if args.mcts_ucb_use_action_prob_for_exploration:
                    action_prob_for_exploration = policy[action]

                if cache_mgr.sa_qval.fn_does_key_exist(state_action_key):
                    parent_visit_factor = state_visits.fn_get_state_visits(state_key)
                    if args.mcts_ucb_use_log_in_numerator:
                        parent_visit_factor = np.log(parent_visit_factor)

                    ucb = cache_mgr.sa_qval.fn_get_data(state_action_key) \
                          + args.cpuct_exploration_exploitation_factor * action_prob_for_exploration * math.sqrt \
                                  (
                                  parent_visit_factor / state_visits.fn_get_child_state_visits(state_action_key)
                              )
                else:
                    ucb = args.cpuct_exploration_exploitation_factor * action_prob_for_exploration * math.sqrt(
                        state_visits.fn_get_state_visits(state_key) + EPS)  # Q = 0 ?

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_act = action
        action = best_act
        return action

    ret_functions = namedtuple('_', [
        'fn_get_visit_counts',
        'fn_get_best_ucb_action',
        'fn_update_state_during_backprop',

        'fn_get_cached_results',
        'fn_visit_new_state_if_possible',
    ])
    # ret_functions.cache_mgr = cache_mgr
    ret_functions.fn_get_visit_counts = fn_get_visit_counts
    ret_functions.fn_get_best_ucb_action = fn_get_best_ucb_action
    ret_functions.fn_update_state_during_backprop = fn_update_state_during_backprop
    # ret_functions.fn_get_cached_predictions = fn_get_cached_predictions

    ret_functions.fn_get_cached_results = fn_get_cached_results
    ret_functions.fn_visit_new_state_if_possible = fn_visit_new_state_if_possible

    return ret_functions
