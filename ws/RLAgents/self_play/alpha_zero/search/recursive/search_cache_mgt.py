from collections import namedtuple


def search_cache_mgt():
    Es = {}     # stores game.fn_get_game_progress_status ended for board_pieces state

    def fn_does_end_exist(state_key):
        return state_key in Es

    def fn_get_end_state(state_key):
        if fn_does_end_exist(state_key):
            return Es[state_key]
        else:
            return None


    def fn_set_end_state(state_key, val):
        nonlocal Es
        if fn_does_end_exist(state_key):
            return False
        else:
            Es[state_key] = val
            return True

    search_cache_mgr = namedtuple('_', [
        'fn_does_end_exist',
        'fn_get_end_state',
        'fn_set_end_state'
    ])
    search_cache_mgr.fn_does_end_exist = fn_does_end_exist
    search_cache_mgr.fn_get_end_state = fn_get_end_state
    search_cache_mgr.fn_set_end_state = fn_set_end_state

    return search_cache_mgr
