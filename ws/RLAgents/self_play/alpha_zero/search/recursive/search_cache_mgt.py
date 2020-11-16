from collections import namedtuple


def search_cache_mgt():

    def state_cache(dict):
        _dict = dict
        _cache_hit_count = 0
        _cache_access_count = 0
        _cache_overwrite_try_count = 0

        def fn_does_data_frame_exist(state_key):
            return state_key in _dict

        def fn_get_data(state_key):
            nonlocal _cache_hit_count, _cache_access_count

            _cache_access_count += 1
            if fn_does_data_frame_exist(state_key):
                _cache_hit_count += 1
                return _dict[state_key]
            else:
                return None

        def fn_set_data(state_key, val):
            nonlocal _dict, _cache_overwrite_try_count

            if fn_does_data_frame_exist(state_key):
                _cache_overwrite_try_count += 1
                return False
            else:
                _dict[state_key] = val
            return True

        def fn_get_stats():
            return {
                '_cache_hit_count': _cache_hit_count,
                '_cache_access_count': _cache_access_count,
                '_cache_overwrite_try_count': _cache_overwrite_try_count,
            }


        return fn_does_data_frame_exist, fn_get_data, fn_set_data, fn_get_stats

    Es = {}     # stores game.fn_get_game_progress_status ended for board_pieces state
    result_cache = state_cache(Es)

    def fn_does_result_for_state_exist(state_key):
        return state_key in Es

    def fn_get_result_for_state(state_key):
        if fn_does_result_for_state_exist(state_key):
            return Es[state_key]
        else:
            return None

    def fn_set_result_for_state(state_key, val):
        nonlocal Es
        if fn_does_result_for_state_exist(state_key):
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
        'result_cache',
        'fn_does_end_exist',
        'fn_get_end_state',
        'fn_set_end_state',

        'fn_do_valid_moves_exist',
        'fn_get_valid_moves',
        'fn_set_valid_moves'
    ])
    search_cache_mgr.result_cache = result_cache
    search_cache_mgr.fn_does_result_for_state_exist = fn_does_result_for_state_exist
    search_cache_mgr.fn_get_result_for_state = fn_get_result_for_state
    search_cache_mgr.fn_set_result_for_state = fn_set_result_for_state

    search_cache_mgr.fn_do_valid_moves_exist = fn_do_valid_moves_exist
    search_cache_mgr.fn_get_valid_moves = fn_get_valid_moves
    search_cache_mgr.fn_set_valid_moves = fn_set_valid_moves

    return search_cache_mgr
