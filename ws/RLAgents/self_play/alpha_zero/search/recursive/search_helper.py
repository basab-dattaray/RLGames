import math
from collections import namedtuple

import numpy as np

from ws.RLAgents.self_play.alpha_zero.search.recursive.dict_cache import dict_cache

def search_helper(
        args,
        game_mgr,
        neural_net_mgr,
):
    EPS = 1e-8

    cache = dict_cache()

    def fn_get_visit_counts(state_key):
        counts = [cache.fn_get_attr_data((state_key, a), 'Nsa')
                  if cache.fn_does_attr_key_exist((state_key, a), 'Nsa') else 0 for a in
                  range(game_mgr.fn_get_action_size())]
        return counts

    def fn_get_cached_allowed_moves(state):
        fn_get_allowed_moves = lambda s: game_mgr.fn_get_valid_moves(s, player=1)

        state_key = game_mgr.fn_get_state_key(state)
        if not cache.fn_does_attr_key_exist(state_key, 'allowed_moves'):
            allowed_moves = fn_get_allowed_moves(state)
            cache.fn_set_attr_data(state_key, 'allowed_moves', allowed_moves)
            return allowed_moves
        else:
            return cache.fn_get_attr_data(state_key, 'allowed_moves')

    def fn_get_cached_results(state):
        fn_get_progress_status = lambda s: game_mgr.fn_get_game_progress_status(s, player=1)

        state_key = game_mgr.fn_get_state_key(state)
        if not cache.fn_does_attr_key_exist(state_key, 'result'):
            cache.fn_set_attr_data(state_key, 'result', fn_get_progress_status(state))
        return cache.fn_get_attr_data(state_key, 'result')

    def fn_visit_new_state_if_possible(state):
        state_key = game_mgr.fn_get_state_key(state)
        if not cache.fn_does_attr_key_exist(state_key, 'policy'):
            # leaf node
            policy, state_val, moves_are_allowed = fn_get_cached_predictions(state)
            if not moves_are_allowed:
                return - state_val

            cache.fn_set_data(
                state_key,
                {
                    'policy': policy,
                    'state_val': state_val,
                    'Na': 0,
                })

            return state_val

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

    def fn_expand_if_needed(state_key, action, state_val):
        state_action_key = (state_key, action)

        if not cache.fn_does_attr_key_exist(state_action_key, 'sa_qval'):  # CREATE NEW STATE-ACTION
            cache.fn_set_attr_data(state_action_key, 'sa_qval',  state_val)
            cache.fn_incr_attr_int(state_action_key, 'Nsa')
            ## state_visits.fn_incr_Ns(state_key)
            cache.fn_incr_attr_int(state_key, 'Ns')

    def fn_update_state_during_backprop(state_key, action, state_val):
        state_action_key = (state_key, action)

        sa_visits = cache.fn_get_attr_data( state_action_key, 'Nsa')
        tmp_val = (sa_visits * cache.fn_get_attr_data(state_action_key, 'sa_qval') + state_val) / (sa_visits + 1)
        cache.fn_set_attr_data(state_action_key, 'sa_qval', tmp_val)

        cache.fn_incr_attr_int(state_action_key, 'Nsa')
        ## state_visits.fn_incr_Ns(state_key)
        cache.fn_incr_attr_int(state_key, 'Ns')

    def fn_get_best_ucb_action(state_key):
        allowed_moves = cache.fn_get_attr_data(state_key, 'allowed_moves')

        best_ucb = -float('inf')
        best_act = None

        action_prob_for_exploration = 1

        # pick the action with the highest upper confidence bound
        for action in range(game_mgr.fn_get_action_size()):

            if allowed_moves[action] != 0:
                s_info = cache.fn_get_data(state_key)

                policy = s_info['state_val']
                state_action_key = (state_key, action)

                if args.mcts_ucb_use_action_prob_for_exploration:
                    action_prob_for_exploration = policy[action]

                if cache.fn_does_attr_key_exist(state_action_key, 'sa_qval'):
                    ## parent_visit_factor = state_visits.fn_get_Ns(state_key)
                    parent_visit_factor = cache.fn_get_attr_data(state_key, 'Ns')

                    if args.mcts_ucb_use_log_in_numerator:
                        parent_visit_factor = np.log(parent_visit_factor)

                    ucb = cache.fn_get_attr_data(state_action_key, 'sa_qval') \
                          + args.cpuct_exploration_exploitation_factor * action_prob_for_exploration * math.sqrt \
                              (
                                  parent_visit_factor / cache.fn_get_attr_data(state_action_key, 'Nsa')
                              )
                else:
                    ## ucb = args.cpuct_exploration_exploitation_factor * action_prob_for_exploration * math.sqrt(
                    ##     state_visits.fn_get_Ns(state_key) + EPS)  # Q = 0 ?
                    ucb = args.cpuct_exploration_exploitation_factor * action_prob_for_exploration * math.sqrt(
                        cache.fn_get_attr_data(state_key, 'Ns', 0) + EPS)  # Q = 0 ?

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
        'fn_expand_if_needed',
    ])
    ret_functions.fn_get_visit_counts = fn_get_visit_counts
    ret_functions.fn_get_best_ucb_action = fn_get_best_ucb_action
    ret_functions.fn_update_state_during_backprop = fn_update_state_during_backprop

    ret_functions.fn_get_cached_results = fn_get_cached_results
    ret_functions.fn_visit_new_state_if_possible = fn_visit_new_state_if_possible
    ret_functions.fn_expand_if_needed = fn_expand_if_needed

    return ret_functions
