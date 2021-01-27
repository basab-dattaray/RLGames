from collections import namedtuple

from ws.RLUtils.algo_lib.search.cache_mgt import cache_mgt


def cache2_mgt():
    cache = cache_mgt()
    def fn_does_attr_key_exist(key, attr):
        if not cache.fn_does_key_exist(key):
            return False

        val = cache.fn_get_data(key)
        if isinstance(val, dict):
            return attr in val.keys()
        else:
            return True

    def fn_get_attr_data(key, attr, default=None):
        if cache.fn_does_key_exist(key):
            val = cache.fn_get_data(key)
            if not attr in val:
                return default
            return val[attr]
        else:
            return default

    def fn_set_attr_data(key, attr, val):
        cache.fn_set_data(key, {attr: val})

    def fn_incr_attr_int(key, attr, strict=False):
        attr_exists = fn_does_attr_key_exist(key, attr)
        if not attr_exists:
            if strict:
                return False
            fn_set_attr_data(key, attr, 0)
        val = fn_get_attr_data(key, attr)

        if not isinstance(val, int):
            if strict:
                return False
            val = 0

        val += 1
        fn_set_attr_data(key, attr, val)
        return True

    ret_obj = namedtuple('state_cache', [
        'fn_does_key_exist',
        'fn_get_data',
        'fn_set_data',

        'fn_does_attr_key_exist',
        'fn_get_attr_data',
        'fn_set_attr_data',
        'fn_incr_attr_int',

        'fn_get_stats',
    ])

    ret_obj.fn_does_key_exist = cache.fn_does_key_exist
    ret_obj.fn_get_data = cache.fn_get_data
    ret_obj.fn_set_data = cache.fn_set_data

    ret_obj.fn_does_attr_key_exist = fn_does_attr_key_exist
    ret_obj.fn_get_attr_data = fn_get_attr_data
    ret_obj.fn_set_attr_data = fn_set_attr_data
    ret_obj.fn_incr_attr_int = fn_incr_attr_int

    ret_obj.fn_get_stats = cache.fn_get_stats

    return ret_obj