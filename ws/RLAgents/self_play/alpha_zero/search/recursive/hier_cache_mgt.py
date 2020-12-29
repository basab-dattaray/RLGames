from collections import namedtuple
from ws.RLAgents.self_play.alpha_zero.search.recursive.dict_cache import dict_cache

def hier_cache_mgt():
    s_info = dict_cache()

    sa_qval = dict_cache()

    ret_obj = namedtuple('_', [
        's_info',
        'sa_qval',

    ])
    ret_obj.s_info = s_info

    ret_obj.sa_qval = sa_qval

    return ret_obj
