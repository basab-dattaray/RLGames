from collections import namedtuple

from ws.RLEnvironments.gridworld import CONFIG
from ws.RLEnvironments.gridworld.grid_board.display_mgt import display_mgt
from ws.RLEnvironments.gridworld.grid_board.policy_repo_mgt import policy_table_mgt

from ws.RLEnvironments.gridworld.grid_board.values_repo_mgt import values_repo_mgt

def env_mgt(name, strategy= None):
    ACTION_SIZE = 4
    config = CONFIG.fn_get_config()

    _board_blockers = config.DISPLAY['BOARD_BLOCKERS']
    _board_goal = config.DISPLAY['BOARD_GOAL']
    _width = config.DISPLAY['WIDTH']
    _height = config.DISPLAY['HEIGHT']

    _reward = None

    _all_states = None
    _current_state = None
    display_mgr = display_mgt(strategy)
    _values_repo_mgr = values_repo_mgt(display_mgr)
    _policy_repo_mgr = policy_table_mgt(ACTION_SIZE, _width, _height)

    def fn_reset_env():
        nonlocal  _reward,  _all_states, _current_state

        _reward = [[0] * _width for _ in range(_height)]

        for blocker in _board_blockers:
            _reward[blocker['y']][blocker['x']] = blocker['reward']  # for square

        _reward[_board_goal['y']][_board_goal['x']] = _board_goal['reward']  # for triangle
        _all_states = []

        for x in range(_width):
            for y in range(_height):
                state = [x, y]
                _all_states.append(state)

        _current_state = [0, 0]
        return _current_state

    def _fn_env_step(action):
        nonlocal _current_state
        one = 1
        next_state_x = _current_state[0]
        next_state_y = _current_state[1]

        if action == 0:  # up
            if _current_state[1] >= one:
                next_state_y -= one
        elif action == 1:  # down
            if _current_state[1] < (_height - 1) * one:
                next_state_y += one
        elif action == 2:  # left
            if _current_state[0] >= one:
                next_state_x -= one
        elif action == 3:  # right
            if _current_state[0] < (_width - 1) * one:
                next_state_x += one

        return next_state_x, next_state_y

    def fn_take_step(action, planning_mode=False):
        nonlocal  _current_state

        next_state = _fn_env_step(action)
        reward = _reward[next_state[1]][next_state[0]]

        if not planning_mode:
            _current_state = next_state

        return next_state, reward, None, None

    def fn_render():
        pass

    def fn_get_state_size():
        return [_width, _height]

    def fn_get_action_size():
        return ACTION_SIZE

    def fn_close():
        display_mgr.fn_close()

    def fn_set_active_state(current_state):
        nonlocal _current_state
        _current_state = current_state

    def fn_get_all_states():
        return _all_states

    def fn_value_table_possible_actions():
        return [0, 1, 2, 3]

    def fn_get_allowed_moves():
        return [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def fn_get_strategy():
        return strategy

    def fn_is_goal_reached(state):
        return True if state == [_board_goal['x'], _board_goal['y']] else False

    fn_get_config = CONFIG.fn_get_config

    fn_reset_env()
    ret_obj = namedtuple('_', [
        'display_mgr',
        'fn_reset_env',
        'fn_take_step',
        'fn_render',
        'fn_get_state_size',
        'fn_get_action_size',
        'fn_close',

        'fn_set_active_state',
        'fn_get_all_states',
        'fn_value_table_possible_actions',
        'fn_get_allowed_moves',
        'fn_get_config',
        'fn_get_strategy',
        'fn_is_goal_reached',
        'values_repo_mgr',
        'policy_repo_mgr',
        'ERROR_MESSAGE',
    ])

    ret_obj.display_mgr = display_mgr
    ret_obj.values_repo_mgr = _values_repo_mgr
    ret_obj.policy_repo_mgr = _policy_repo_mgr

    ret_obj.fn_reset_env = fn_reset_env
    ret_obj.fn_take_step = fn_take_step
    ret_obj.fn_render = fn_render
    ret_obj.fn_get_state_size = fn_get_state_size
    ret_obj.fn_get_action_size = fn_get_action_size
    ret_obj.fn_close = fn_close

    ret_obj.fn_set_active_state = fn_set_active_state
    ret_obj.fn_get_all_states = fn_get_all_states
    ret_obj.fn_value_table_possible_actions = fn_value_table_possible_actions
    ret_obj.fn_get_allowed_moves = fn_get_allowed_moves
    ret_obj.fn_get_config = fn_get_config
    ret_obj.fn_get_strategy = fn_get_strategy
    ret_obj.fn_is_goal_reached = fn_is_goal_reached

    ret_obj.ERROR_MESSAGE = None

    return ret_obj, None
