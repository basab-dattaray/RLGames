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

    def _fn_get_valid_moves(state, player):
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
        valid_moves = _fn_get_valid_moves(state, player)
        if valid_moves is None:
            return policy, state_val, valid_moves
        policy = policy * valid_moves  # masking invalid moves
        sum_Ps_s = numpy.sum(policy)
        if sum_Ps_s > 0:
            policy /= sum_Ps_s  # renormalize
        else:
            policy = policy + valid_moves
            policy /= numpy.sum(policy)
        return policy, state_val, valid_moves

    def fn_find_best_ucb_child(state, children_nodes, visits, explore_exploit_ratio):
        best_child = None
        best_ucb = 0

        # policy, _, _ = fn_get_prediction_info(state)
        policy, state_val = _fn_get_state_predictions(state)

        for key, child_node in children_nodes.items():
            action_num = int(key)
            action_prob = policy[action_num]

            child_visits = child_node.fn_get_num_visits()
            child_value = child_node.fn_get_node_val()
            if child_visits == 0:
                return child_node

            exploit_val = child_value / child_visits
            explore_val = action_prob * math.sqrt(visits) / (child_visits + 1)
            ucb = exploit_val + explore_exploit_ratio * explore_val  # Upper Confidence Bound

            if best_child is None:
                best_child = child_node
                best_ucb = ucb
            else:
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child_node

        return best_child

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




    return fn_get_prediction_info, fn_find_best_ucb_child