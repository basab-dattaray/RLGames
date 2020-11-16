from collections import namedtuple


def search_cache_mgt():

    def state_cache(dict):
        def _fn_does_DATA_exist(state_key):
            return state_key in dict

        def _fn_get_DATA(state_key):
            if _fn_does_DATA_exist(state_key):
                return dict[state_key]
            else:
                return None

        def _fn_set_DATA(state_key, val):
            nonlocal dict
            if _fn_does_DATA_exist(state_key):
                return False
            else:
                dict[state_key] = val
            return True

    Es = {}     # stores game.fn_get_game_progress_status ended for board_pieces state

    def fn_does_end_state_exist(state_key):
        return state_key in Es

    def fn_get_end_state(state_key):
        if fn_does_end_state_exist(state_key):
            return Es[state_key]
        else:
            return None

    def fn_set_end_state(state_key, val):
        nonlocal Es
        if fn_does_end_state_exist(state_key):
            return False
        else:
            Es[state_key] = val
            return True
    #
    Vs = {}  # stores game.fn_get_valid_moves for board_pieces state

    def fn_do_valid_moves_exist(state_key):
        return state_key in Vs

    def fn_get_valid_moves(state_key):
        if fn_do_valid_moves_exist(state_key):
            return Vs[state_key]
        else:
            return None

    def fn_set_valid_moves(state_key, val):
        nonlocal Vs
        if fn_do_valid_moves_exist(state_key):
            return False
        else:
            Vs[state_key] = val
            return True

    search_cache_mgr = namedtuple('_', [
        'fn_does_end_exist',
        'fn_get_end_state',
        'fn_set_end_state',

        'fn_do_valid_moves_exist',
        'fn_get_valid_moves',
        'fn_set_valid_moves'
    ])
    search_cache_mgr.fn_does_end_state_exist = fn_does_end_state_exist
    search_cache_mgr.fn_get_end_state = fn_get_end_state
    search_cache_mgr.fn_set_end_state = fn_set_end_state

    search_cache_mgr.fn_do_valid_moves_exist = fn_do_valid_moves_exist
    search_cache_mgr.fn_get_valid_moves = fn_get_valid_moves
    search_cache_mgr.fn_set_valid_moves = fn_set_valid_moves

    return search_cache_mgr
