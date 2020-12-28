from collections import namedtuple
# from ws.RLAgents.self_play.alpha_zero.search.dictionary_cache import dictionary_cache
from ws.RLAgents.self_play.alpha_zero.search.recursive.dict_cache import dict_cache


def hier_cache_mgt():
    s_results = dict_cache()
    s_allowed_moves = dict_cache()
    s_predictions = dict_cache()

    sa_qval = dict_cache()

    ret_obj = namedtuple('_', [
        's_results',
        's_predictions',
        's_allowed_moves',

        'sa_qval',

    ])
    ret_obj.s_results = s_results
    ret_obj.s_allowed_moves = s_allowed_moves
    ret_obj.s_predictions = s_predictions

    ret_obj.sa_qval = sa_qval
    return ret_obj
