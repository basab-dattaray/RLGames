from collections import namedtuple

from ws.RLEnvironments.gridworld.logic.SETUP_INFO import POSSIBLE_ACTIONS

from .logic.episode_mgt import episode_mgt

def env_mgt(app_info):

    # _app_info = app_info
    app_info['OBJ_EPISODE'] = episode_mgt()

    _board_blockers = app_info["display"]["BOARD_BLOCKERS"]
    _board_goal = app_info["display"]["BOARD_GOAL"]
    _width = app_info["display"]["WIDTH"]
    _height = app_info["display"]["HEIGHT"]

    _reward = None
    # _possible_actions = None
    _all_states = None
    _state = []

    def fn_reset_env():
        nonlocal  _reward,  _all_states, _state

        app_info['OBJ_EPISODE'] = episode_mgt()

        _reward = [[0] * _width for _ in range(_height)]
        _possible_actions = POSSIBLE_ACTIONS

        for blocker in _board_blockers:
            _reward[blocker['y']][blocker['x']] = blocker['reward']  # for square

        _reward[_board_goal['y']][_board_goal['x']] = _board_goal['reward']  # for triangle
        _all_states = []

        for x in range(_width):
            for y in range(_height):
                state = [x, y]
                _all_states.append(state)

        _state = [0, 0]
        return _state

    def _fn_env_step(action):
        nonlocal _state
        one = 1
        next_state_x = _state[0]
        next_state_y = _state[1]

        if action == 0:  # up
            if _state[1] >= one:
                next_state_y -= one
        elif action == 1:  # down
            if _state[1] < (_height - 1) * one:
                next_state_y += one
        elif action == 2:  # left
            if _state[0] >= one:
                next_state_x -= one
        elif action == 3:  # right
            if _state[0] < (_width - 1) * one:
                next_state_x += one

        return next_state_x, next_state_y

    def fn_take_step(action, planning_mode=False):
        nonlocal  _state

        next_state = _fn_env_step(action)
        reward = _reward[next_state[1]][next_state[0]]

        # app_info['OBJ_EPISODE'].fn_update_episode(_reward[next_state[1]][next_state[0]])

        if not planning_mode:
            _state = next_state

        return next_state, reward, None, None

    def fn_render():
        pass

    def fn_get_state_size():
        return [_width, _height]

    def fn_get_action_size():
        return [2]

    def fn_close():
        pass

    def fn_update_state(state):
        nonlocal _state
        _state = state

    def fn_get_all_states():
        return _all_states

    def fn_value_table_possible_actions():
        return POSSIBLE_ACTIONS

    fn_reset_env()
    ret_obj = namedtuple('_', [
        'fn_reset_env',
        'fn_take_step',
        'fn_render',
        'fn_get_state_size',
        'fn_get_action_size',
        'fn_close',

        'fn_update_state',
        'fn_get_all_states',
        'fn_value_table_possible_actions',
    ])

    ret_obj.fn_reset_env = fn_reset_env
    ret_obj.fn_take_step = fn_take_step
    ret_obj.fn_render = fn_render
    ret_obj.fn_get_state_size = fn_get_state_size
    ret_obj.fn_get_action_size = fn_get_action_size
    ret_obj.fn_close = fn_close

    ret_obj.fn_update_state = fn_update_state
    ret_obj.fn_get_all_states = fn_get_all_states
    ret_obj.fn_value_table_possible_actions = fn_value_table_possible_actions

    return ret_obj
