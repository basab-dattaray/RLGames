from collections import namedtuple


def search_cache_mgt():

    def state_cache():
        dict = {}
        hit_count = 0
        access_count = 0
        overwrite_try_count = 0

        def fn_does_state_exist(state_key):
            return state_key in dict

        def fn_get_data(state_key):
            nonlocal hit_count, access_count

            access_count += 1
            if fn_does_state_exist(state_key):
                hit_count += 1
                return dict[state_key]
            else:
                return None

        def fn_set_data(state_key, val):
            nonlocal dict, overwrite_try_count

            if fn_does_state_exist(state_key):
                overwrite_try_count += 1
                return False
            else:
                dict[state_key] = val
            return True

        def fn_get_stats():
            return {
                'size': len(dict),
                'hit_count': hit_count,
                'access_count': access_count,
                'overwrite_try_count': overwrite_try_count,
            }

        state_cache = namedtuple('state_cache', [
            'fn_does_state_exist',
            'fn_get_data',
            'fn_set_data',
            'fn_get_stats'
        ])

        state_cache.fn_does_state_exist = fn_does_state_exist
        state_cache.fn_get_data = fn_get_data
        state_cache.fn_set_data = fn_set_data
        state_cache.fn_get_stats = fn_get_stats

        return state_cache

    state_results = state_cache()
    state_valid_moves = state_cache()
    state_policy = state_cache()


    #
    # Vs = {}  # stores game.fn_get_valid_moves for board_pieces state
    #
    # def fn_do_valid_moves_exist(state_key):
    #     return state_key in Vs
    #
    # def fn_get_valid_moves(state_key):
    #     if fn_do_valid_moves_exist(state_key):
    #         return Vs[state_key]
    #     else:
    #         return None
    #
    # def fn_set_valid_moves(state_key, val):
    #     nonlocal Vs
    #     if fn_do_valid_moves_exist(state_key):
    #         return False
    #     else:
    #         Vs[state_key] = val
    #         return True

    search_cache_mgr = namedtuple('_', [
        'result_cache',
        'state_valid_moves',
        'state_policy'

    ])
    search_cache_mgr.state_results = state_results
    search_cache_mgr.state_valid_moves = state_valid_moves
    search_cache_mgr.state_policy = state_policy
    # search_cache_mgr.fn_does_result_for_state_exist = fn_does_result_for_state_exist
    # search_cache_mgr.fn_get_result_for_state = fn_get_result_for_state
    # search_cache_mgr.fn_set_result_for_state = fn_set_result_for_state

    # search_cache_mgr.fn_do_valid_moves_exist = fn_do_valid_moves_exist
    # search_cache_mgr.fn_get_valid_moves = fn_get_valid_moves
    # search_cache_mgr.fn_set_valid_moves = fn_set_valid_moves

    return search_cache_mgr
