
from collections import namedtuple


def cache_mgt():
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

    def fn_set_data(key, val):
        nonlocal dict, overwrite_try_count

        def _fn_set_val(dictionary, key, new_val):
            if type(new_val) == type(dict) and len(dictionary) > 0:

                existing_val = dictionary[key]
                copy_of_dict = existing_val.copy()

                copy_of_dict.update(new_val)
                dictionary[key] = copy_of_dict
            else:
                dictionary[key] = new_val

        if fn_does_key_exist(key):
            overwrite_try_count += 1
            _fn_set_val(dict, key, val)
            return dict
        else:
            dict[key] = val
            return dict

    def fn_does_attr_key_exist(key, attr):
        if not fn_does_key_exist(key):
            return False

        val = fn_get_data(key)
        if type(val) == type(dict):
            return attr in val.keys()
        else:
            return True

    def fn_get_attr_data(key, attr, default= None):
        if fn_does_key_exist(key):
            val = fn_get_data(key)
            if not attr in val:
                return default
            return val[attr]
        else:
            return default

    def fn_set_attr_data(key, attr, val):
        fn_set_data(key, {attr: val})

    def fn_incr_attr_int(key, attr, strict = False):
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

    def fn_get_stats():
        return {
            'size': len(dict),
            'hit_count': hit_count,
            'access_count': access_count,
            'overwrite_try_count': overwrite_try_count,
        }

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

    ret_obj.fn_does_key_exist = fn_does_key_exist
    ret_obj.fn_get_data = fn_get_data
    ret_obj.fn_set_data = fn_set_data

    ret_obj.fn_does_attr_key_exist = fn_does_attr_key_exist
    ret_obj.fn_get_attr_data = fn_get_attr_data
    ret_obj.fn_set_attr_data = fn_set_attr_data
    ret_obj.fn_incr_attr_int = fn_incr_attr_int

    ret_obj.fn_get_stats = fn_get_stats

    return ret_obj