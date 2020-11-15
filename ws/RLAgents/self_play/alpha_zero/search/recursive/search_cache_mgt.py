from collections import namedtuple


def search_cache_mgt():
    Es = {}

    def does_end_exist(state_key):
        return state_key in Es

    def get_end_state(state_key):
        if does_end_exist(state_key):
            return Es[state_key]
        else:
            return None


    def set_end_state(state_key, val):
        nonlocal Es
        if does_end_exist(state_key):
            return False
        else:
            Es[state_key] = val
            return True

    search_cache_mgr = namedtuple('_', [
        'fn_does_end_exist',
        'fn_get_end_state',
        'fn_set_end_state'
    ])
    search_cache_mgr.fn_does_end_exist = does_end_exist
    search_cache_mgr.fn_get_end_state = get_end_state
    search_cache_mgr.fn_set_end_state = set_end_state

    return search_cache_mgr
