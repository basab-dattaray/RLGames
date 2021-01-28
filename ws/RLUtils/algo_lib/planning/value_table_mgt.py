import copy

def value_table_mgt(env):
    LOW_NUMBER = -999999
    env_config = env.fn_get_config()
    # _goal_coordinates = None
    _width = env_config.DISPLAY['WIDTH']
    _height = env_config.DISPLAY['HEIGHT']
    _board_goal =  env_config.DISPLAY['BOARD_GOAL']

    _goal_coordinates = {'x': _board_goal['x'], 'y': _board_goal['y']}
    _value_table = [[0.0] * _width for _ in range(_width)]
    _prev_value_table = None

    def fn_set_state_site_value(state_site, value):
        nonlocal _value_table
        _value_table[state_site[1]][state_site[0]] = value

    def fn_get_state_site_value(state_site):
        return _value_table[state_site[1]][state_site[0]]


    def fn_set_all_state_site_values(table):
        nonlocal _value_table
        _value_table = table

    def fn_get_all_state_site_values():
        return _value_table

    def fn_has_any_state_site_changed():
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

    def fn_get_state_site_actions(state_site):

        row, col = state_site
        action_size = env.fn_get_action_size()
        possible_actions = [LOW_NUMBER] * action_size

        dir_up = [row, max(0, col -1)]
        if col > 0:
            possible_actions[0] = fn_get_state_site_value(dir_up)

        dir_down = [row, min(_height - 1, col + 1) ]
        if col < _height - 1:
            possible_actions[1] = fn_get_state_site_value(dir_down)

        dir_left = [max(0, row - 1), col]
        if row > 0:
            possible_actions[2] = fn_get_state_site_value(dir_left)

        dir_right = [ min(_width - 1, row + 1) , col]
        if row < _width - 1:
            possible_actions[3] = fn_get_state_site_value(dir_right)

        return possible_actions

    def fn_goal_reached(state_site):
        return True if state_site == [_goal_coordinates['x'], _goal_coordinates['y']] else False

    return fn_set_state_site_value, fn_get_state_site_value, fn_set_all_state_site_values, fn_get_all_state_site_values, fn_get_state_site_actions, fn_goal_reached, fn_has_any_state_site_changed