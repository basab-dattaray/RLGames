from collections import namedtuple

from ws.RLAgents.self_play.alpha_zero.search.dictionary_cache import dictionary_cache


def cache_mgt():

    state_results = dictionary_cache()
    state_valid_moves = dictionary_cache()
    state_policy = dictionary_cache()
    state_action_qval = dictionary_cache()
    state_info = dictionary_cache()
    state_predictions = dictionary_cache()

    cache_mgr = namedtuple('_', [
        'state_results',
        'state_valid_moves',
        'state_policy',
        'state_action_qval',
        'state_info',
        'state_predictions'

    ])
    cache_mgr.state_results = state_results
    cache_mgr.state_valid_moves = state_valid_moves
    cache_mgr.state_policy = state_policy
    cache_mgr.state_action_qval = state_action_qval
    cache_mgr.state_info = state_info
    cache_mgr.state_predictions = state_predictions

    return cache_mgr
