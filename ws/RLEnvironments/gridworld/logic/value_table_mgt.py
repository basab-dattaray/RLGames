import copy
from ws.RLEnvironments.gridworld.logic.SETUP_INFO import ACTION_MOVE_STATE_RULES

def value_table_mgr(app_info):
    LOW_NUMBER = -999999
    _goal_coordinates = None
    app_info_display = app_info['display']
    _width = app_info_display["WIDTH"]
    _height = app_info_display['HEIGHT']
    _board_goal = app_info_display["BOARD_GOAL"]

    _goal_coordinates = {'x': _board_goal['x'], 'y': _board_goal['y']}
    _value_table = [[0.0] * _width for _ in range(_width)]
    _prev_value_table = None

    def fn_set_value_table_item(state, value):
        nonlocal _value_table
        _value_table[state[1]][state[0]] = value

    def fn_get_value_table_item(state):
        return _value_table[state[1]][state[0]]


    def fn_set_value_table(table):
        nonlocal _value_table
        _value_table = table

    def fn_get_value_table():
        return _value_table

    def fn_has_table_changed():
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

    def fn_value_table_possible_actions(state):

        row, col = state
        possible_actions = [LOW_NUMBER] * len(ACTION_MOVE_STATE_RULES)

        vt = _value_table

        dir_up = [row, max(0, col -1)]
        if col > 0:
            possible_actions[0] = fn_get_value_table_item(dir_up)

        dir_down = [row, min(_height - 1, col + 1) ]
        if col < _height - 1:
            possible_actions[1] = fn_get_value_table_item(dir_down)

        dir_left = [max(0, row - 1), col]
        if row > 0:
            possible_actions[2] = fn_get_value_table_item(dir_left)

        dir_right = [ min(_width - 1, row + 1) , col]
        if row < _width - 1:
            possible_actions[3] = fn_get_value_table_item(dir_right)

        return possible_actions

    def fn_value_table_reached_target(state):
        return True if state == [_goal_coordinates['x'], _goal_coordinates['y']] else False

    return fn_set_value_table_item, fn_get_value_table_item, fn_set_value_table, fn_get_value_table, fn_value_table_possible_actions,fn_value_table_reached_target, fn_has_table_changed