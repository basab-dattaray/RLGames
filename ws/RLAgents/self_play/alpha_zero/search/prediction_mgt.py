from collections import namedtuple

import numpy


def prediction_mgt(game_mgr, cache_mgr, neural_net_mgr):
    def _fn_get_state_predictions(state):
        state_key = game_mgr.fn_get_state_key(state)

        predictions = cache_mgr.state_predictions.fn_get_data_or_none(state_key)
        if predictions is not None:
            return predictions

        action_probalities, wrapped_state_val = neural_net_mgr.predict(state)
        predictions = (action_probalities, wrapped_state_val[0])
        cache_mgr.state_predictions.fn_set_data(state_key, predictions)
        return predictions

    def _fn_get_valid_moves(state, player):
        state_key = game_mgr.fn_get_state_key(state)

        valids = cache_mgr.state_valid_moves.fn_get_data_or_none(state_key)
        if valids is not None:
            return valids

        valids = game_mgr.fn_get_valid_moves(state, player)
        if valids[-1] == 1:
            return None
        cache_mgr.state_valid_moves.fn_set_data(state_key, valids)

        return valids

    def fn_get_prediction_info(state, player):
        action_probalities, state_val = _fn_get_state_predictions(state)
        valid_moves = _fn_get_valid_moves(state, player)

        action_probalities = action_probalities * valid_moves  # masking invalid moves
        sum_Ps_s = numpy.sum(action_probalities)
        if sum_Ps_s > 0:
            action_probalities /= sum_Ps_s  # renormalize
        else:
            action_probalities = action_probalities + valid_moves
            action_probalities /= numpy.sum(action_probalities)
        return action_probalities, state_val, valid_moves

    return fn_get_prediction_info