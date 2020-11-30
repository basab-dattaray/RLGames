from collections import namedtuple


def dictionary_cache():
    dict = {}
    hit_count = 0
    access_count = 0
    overwrite_try_count = 0

    def fn_does_key_exist(key):
        return key in dict

    def fn_get_data(key):
        nonlocal hit_count, access_count

        access_count += 1
        hit_count += 1
        return dict[key]


    def fn_get_data_or_none(key):
        nonlocal hit_count, access_count

        access_count += 1
        if fn_does_key_exist(key):
            hit_count += 1
            return dict[key]
        else:
            return None

    def fn_set_data(key, val):
        nonlocal dict, overwrite_try_count

        def _fn_set_val(dict, key, val):
            if type(val) == dict:
                revised_dict = dict[key].copy()
                revised_dict.update(val)
                dict[key] = revised_dict
            else:
                dict[key] = val


        if fn_does_key_exist(key):
            overwrite_try_count += 1
            _fn_set_val(dict, key, val)
            return False
        else:
            dict[key] = val
            return True

    def fn_get_stats():
        return {
            'size': len(dict),
            'hit_count': hit_count,
            'access_count': access_count,
            'overwrite_try_count': overwrite_try_count,
        }

    dictionary_cache = namedtuple('state_cache', [
        'fn_does_key_exist',
        'fn_get_data',
        'fn_get_data_or_none',
        'fn_set_data',
        'fn_get_stats'
    ])

    dictionary_cache.fn_does_key_exist = fn_does_key_exist
    dictionary_cache.fn_get_data = fn_get_data
    dictionary_cache.fn_get_data_or_none = fn_get_data_or_none
    dictionary_cache.fn_set_data = fn_set_data
    dictionary_cache.fn_get_stats = fn_get_stats

    return dictionary_cache
