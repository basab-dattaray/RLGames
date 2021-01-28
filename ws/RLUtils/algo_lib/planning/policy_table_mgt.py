from collections import namedtuple


def policy_table_mgt(env):
    config = env.fn_get_config()
    init_policy = [1/ env.fn_get_action_size()] * env.fn_get_action_size() # [.25, .25, .25, .25]
    _policy_table = [init_policy * config.DISPLAY['WIDTH'] for _ in range(config.DISPLAY['HEIGHT'])]

    def fn_get_policy_state_value(state):
        return _policy_table[state[1]][state[0]]

    def fn_set_policy_state_value(state, value):
        nonlocal _policy_table
        _policy_table[state[1]][state[0]] = value

    def fn_fetch_policy_table():
        return _policy_table

    ret_obj = namedtuple('_', [
        'fn_get_policy_state_value',
        'fn_set_policy_state_value',
        'fn_fetch_policy_table',
    ])

    ret_obj.fn_get_policy_state_value = fn_get_policy_state_value
    ret_obj.fn_set_policy_state_value = fn_set_policy_state_value
    ret_obj.fn_fetch_policy_table = fn_fetch_policy_table

    return ret_obj