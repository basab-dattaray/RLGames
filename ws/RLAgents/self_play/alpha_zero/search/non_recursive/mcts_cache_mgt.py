from collections import namedtuple

def mcts_cache_mgt(
    fn_get_state_key,
    fn_terminal_value
):
    progress_status_dict = {}
    def fn_get_progress_status(state):
        state_key = fn_get_state_key(state)

        if state_key not in progress_status_dict:
            progress_status_dict[state_key] = fn_terminal_value(state)

        return progress_status_dict[state_key]


    mcts_cache_mgr = namedtuple('_', ['fn_get_progress_status'])
    mcts_cache_mgr.fn_get_progress_status = fn_get_progress_status

    return mcts_cache_mgr