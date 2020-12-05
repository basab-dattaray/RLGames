import math

import numpy

def cache2_mgt(game_mgr, cache_mgr, neural_net_mgr):
    EPS = 1e-8
    
    def _fn_get_state_predictions(state):
        state_key = game_mgr.fn_get_state_key(state)

        predictions = cache_mgr.state_predictions.fn_get_data_or_none(state_key)
        if predictions is not None:
            return predictions

        policy, wrapped_state_val = neural_net_mgr.predict(state)
        predictions = (policy, wrapped_state_val[0])
        cache_mgr.state_predictions.fn_set_data(state_key, predictions)
        return predictions

    def fn_get_valid_moves(state, player):
        state_key = game_mgr.fn_get_state_key(state)

        valid_moves = cache_mgr.state_valid_moves.fn_get_data_or_none(state_key)
        if valid_moves is not None:
            return valid_moves

        valid_moves = game_mgr.fn_get_valid_moves(state, player)
        if valid_moves is None:
            return None
        cache_mgr.state_valid_moves.fn_set_data(state_key, valid_moves)

        return valid_moves

    def fn_get_prediction_info(state, player):
        policy, state_val = _fn_get_state_predictions(state)
        valid_moves = fn_get_valid_moves(state, player)
        if valid_moves is None:
            return policy, state_val, valid_moves
        policy = policy * valid_moves  # masking invalid moves
        sum_policy = numpy.sum(policy)
        if sum_policy > 0:
            policy /= sum_policy  # renormalize
        else:
            policy = policy + valid_moves
            policy /= numpy.sum(policy)
        return policy, state_val, valid_moves


    #
    # def fn_get_best_ucb_action(state, max_num_actions, explore_exploit_ratio):
    #     state_key = game_mgr.fn_get_state_key(state)
    #     valid_moves = cache_mgr.state_valid_moves.fn_get_data(state_key)
    #
    #     best_ucb = -float('inf')
    #     best_act = -1
    #     # pick the action with the highest upper confidence bound
    #     policy, state_val = _fn_get_state_predictions(state)
    #
    #     for action in range(max_num_actions):
    #         if valid_moves[action]:
    #             # policy = state_policy.fn_get_data(state_key)
    #             state_action_key = (state_key, action)
    #
    #             if state_action_qval.fn_does_key_exist(state_action_key):
    #                 # qval = cache_mgr.state_action_qval.fn_get_data(key)  # Qsa[(state_key, action)]
    #                 ucb = state_action_qval.fn_get_data(state_action_key) \
    #                       + explore_exploit_ratio * policy[action] * numpy.math.sqrt(
    #                     numpy.log(state_visits.fn_get_state_visits(state_key)) /
    #                           state_visits.fn_get_child_state_visits(state_action_key))
    #
    #             else:
    #                 ucb = explore_exploit_ratio * policy[action] * numpy.math.sqrt(
    #                     state_visits.fn_get_state_visits(state_key) + EPS)  # Q = 0 ?
    #                 # u = 0
    #             if ucb > best_ucb:
    #                 best_ucb = ucb
    #                 best_act = action
    #     action = best_act
    #     return action




    return fn_get_prediction_info, fn_find_best_ucb_child, fn_get_valid_moves