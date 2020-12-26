from collections import namedtuple

from ws.RLAgents.self_play.alpha_zero.search.dictionary_cache import dictionary_cache


def cache_mgt():

    s_results = dictionary_cache()
    s_allowed_moves = dictionary_cache()
    s_predictions = dictionary_cache()

    sa_qval = dictionary_cache()

    cache_mgr = namedtuple('_', [
        's_results',
        's_predictions',
        's_allowed_moves',

        'sa_qval',

    ])
    cache_mgr.s_results = s_results
    cache_mgr.s_allowed_moves = s_allowed_moves
    cache_mgr.s_predictions = s_predictions

    cache_mgr.sa_qval = sa_qval
    return cache_mgr
