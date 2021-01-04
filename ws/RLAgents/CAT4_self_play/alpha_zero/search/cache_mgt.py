
from collections import namedtuple


def cache_mgt():
    dictionary = {}
    hit_count = 0
    access_count = 0
    overwrite_try_count = 0

    def fn_does_key_exist(key):
        return key in dictionary

    def fn_get_data(key):
        nonlocal hit_count, access_count

        access_count += 1
        hit_count += 1
        return dictionary[key]

    def fn_set_data(key, val):
        nonlocal dictionary, overwrite_try_count

        def _fn_set_val(dictionary, key, new_val):
            if isinstance(new_val, dict) and len(dictionary) > 0:

                existing_val = dictionary[key]
                copy_of_dict = existing_val.copy()

                copy_of_dict.update(new_val)
                dictionary[key] = copy_of_dict
            else:
                dictionary[key] = new_val

        if fn_does_key_exist(key):
            overwrite_try_count += 1
            _fn_set_val(dictionary, key, val)
            return dictionary
        else:
            dictionary[key] = val
            return dictionary


    def fn_get_stats():
        return {
            'size': len(dictionary),
            'hit_count': hit_count,
            'access_count': access_count,
            'overwrite_try_count': overwrite_try_count,
        }

    ret_obj = namedtuple('_', [
        'fn_does_key_exist',
        'fn_get_data',
        'fn_set_data',

        'fn_get_stats',
    ])

    ret_obj.fn_does_key_exist = fn_does_key_exist
    ret_obj.fn_get_data = fn_get_data
    ret_obj.fn_set_data = fn_set_data

    ret_obj.fn_get_stats = fn_get_stats

    return ret_obj
