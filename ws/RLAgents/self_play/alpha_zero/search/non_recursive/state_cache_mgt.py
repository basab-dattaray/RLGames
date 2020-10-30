from collections import namedtuple

import numpy

def state_cache_mgt(fn_get_valid_actions, fn_predict_action_probablities, state):

    valid_norm_action_probabilities = None
    action_probabilities = None
    value = None

    def __fn_get_valid_normalized_action_probabilities_impl(action_probabilities):
        valid_moves = fn_get_valid_actions(state)
        if valid_moves is None:
            return None
        valid_action_probabilities = action_probabilities * valid_moves
        sum_action_probs = numpy.sum(valid_action_probabilities)
        if sum_action_probs > 0:
            return valid_action_probabilities / sum_action_probs
        else:
            distributed_action_probabilities = [1/len(action_probabilities)] * len(action_probabilities)
            return distributed_action_probabilities

    def fn_get_valid_normalized_action_probabilities():
        action_probabilities = None
        nonlocal valid_norm_action_probabilities
        if valid_norm_action_probabilities is None:
            if action_probabilities is None:
                action_probabilities, _ = fn_get_predictions()
            valid_norm_action_probabilities = __fn_get_valid_normalized_action_probabilities_impl(action_probabilities)
        return valid_norm_action_probabilities

    def fn_get_predictions():
        nonlocal action_probabilities, value
        if action_probabilities is None or value is None:
            action_probabilities, value = fn_predict_action_probablities(state)
        return action_probabilities, value

    state_cache_mgr = namedtuple('_', ['fn_get_valid_normalized_action_probabilities', 'fn_get_predictions'])
    state_cache_mgr.fn_get_valid_normalized_action_probabilities = fn_get_valid_normalized_action_probabilities
    state_cache_mgr.fn_get_predictions = fn_get_predictions

    return state_cache_mgr
