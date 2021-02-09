import copy
from collections import namedtuple


def values_repo_mgt(Display):

    _prev_value_table = None
    _value_table = Display.fn_create_value_repo()

    def fn_set_state_value(state, value):
        nonlocal _value_table
        _value_table[state[1]][state[0]] = value

    def fn_get_state_value(state):
        return _value_table[state[1]][state[0]]

    def fn_set_all_state_values(table):
        nonlocal _value_table
        _value_table = table

    def fn_get_all_state_values():
        return _value_table

    def fn_has_state_changed():
        nonlocal _prev_value_table

        if _prev_value_table is None:
            _prev_value_table = copy.deepcopy(_value_table)
            return True

        return Display.fn_compare_value_repos(_prev_value_table, _value_table)

    ret_obj = namedtuple('_', [
        'fn_set_state_value',
        'fn_get_state_value',
        'fn_set_all_state_values',
        'fn_has_any_state_changed',
    ])

    ret_obj.fn_set_state_value = fn_set_state_value
    ret_obj.fn_get_state_value = fn_get_state_value
    ret_obj.fn_set_all_state_values = fn_set_all_state_values

    ret_obj.fn_get_all_state_values = fn_get_all_state_values
    ret_obj.fn_has_any_state_changed = fn_has_state_changed

    return ret_obj
