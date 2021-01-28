import copy
from collections import namedtuple


def values_repo_mgt(env):
    LOW_NUMBER = -999999
    env_config = env.fn_get_config()
    # _goal_coordinates = None
    _width = env_config.DISPLAY['WIDTH']
    _height = env_config.DISPLAY['HEIGHT']
    _board_goal =  env_config.DISPLAY['BOARD_GOAL']

    _goal_coordinates = {'x': _board_goal['x'], 'y': _board_goal['y']}
    _value_table = [[0.0] * _width for _ in range(_width)]
    _prev_value_table = None

    def fn_set_statepart_value(statepart, value):
        nonlocal _value_table
        _value_table[statepart[1]][statepart[0]] = value

    def fn_get_statepart_value(statepart):
        return _value_table[statepart[1]][statepart[0]]


    def fn_set_all_statepart_values(table):
        nonlocal _value_table
        _value_table = table

    def fn_get_all_statepart_values():
        return _value_table

    def fn_has_any_statepart_changed():
        nonlocal _prev_value_table

        if _prev_value_table is None:
            _prev_value_table = copy.deepcopy(_value_table)
            return True
        # if value_table == None:
        #     return True
        for col in range(0, _height):
            for row in range(0, _width):
                if _prev_value_table[col][row] != _value_table[col][row]:
                    _prev_value_table[col][row] = _value_table[col][row]
                    return True
        return False

    def fn_get_statepart_actions(statepart):

        row, col = statepart
        action_size = env.fn_get_action_size()
        possible_actions = [LOW_NUMBER] * action_size

        dir_up = [row, max(0, col -1)]
        if col > 0:
            possible_actions[0] = fn_get_statepart_value(dir_up)

        dir_down = [row, min(_height - 1, col + 1) ]
        if col < _height - 1:
            possible_actions[1] = fn_get_statepart_value(dir_down)

        dir_left = [max(0, row - 1), col]
        if row > 0:
            possible_actions[2] = fn_get_statepart_value(dir_left)

        dir_right = [ min(_width - 1, row + 1) , col]
        if row < _width - 1:
            possible_actions[3] = fn_get_statepart_value(dir_right)

        return possible_actions

    ret_obj = namedtuple('_', [
        'fn_set_statepart_value',
        'fn_get_statepart_value',
        'fn_set_all_statepart_values',
        'fn_get_statepart_actions',
        'fn_has_any_statepart_changed',
    ])

    ret_obj.fn_set_statepart_value = fn_set_statepart_value
    ret_obj.fn_get_statepart_value = fn_get_statepart_value
    ret_obj.fn_set_all_statepart_values = fn_set_all_statepart_values

    ret_obj.fn_get_all_statepart_values = fn_get_all_statepart_values
    ret_obj.fn_get_statepart_actions = fn_get_statepart_actions
    ret_obj.fn_has_any_statepart_changed = fn_has_any_statepart_changed

    return ret_obj


    # return fn_set_statepart_value, fn_get_statepart_value, fn_set_all_statepart_values, fn_get_all_statepart_values, fn_get_statepart_actions, fn_has_any_statepart_changed