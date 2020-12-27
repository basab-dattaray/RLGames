from collections import namedtuple
# from ws.RLAgents.self_play.alpha_zero.search.dictionary_cache import dictionary_cache
from ws.RLAgents.self_play.alpha_zero.search.recursive.dict_cache import dict_cache


def hier_cache_mgt():
    s_results = dict_cache()
    s_allowed_moves = dict_cache()
    s_predictions = dict_cache()

    sa_qval = dict_cache()

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
