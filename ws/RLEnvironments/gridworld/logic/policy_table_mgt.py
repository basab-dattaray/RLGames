from ws.RLEnvironments.gridworld.logic.SETUP_INFO import INITIAL_ACTION_PROBABILITIES

def policy_table_mgr(app_info):
    LOW_NUMBER = -999999
    _policy_table = [[INITIAL_ACTION_PROBABILITIES] * app_info['display']["WIDTH"] for _ in range(app_info['display']["HEIGHT"])]

    def fn_get_policy_state_value(state):
        return _policy_table[state[1]][state[0]]

    def fn_set_policy_state_value(state, value):
        nonlocal _policy_table
        _policy_table[state[1]][state[0]] = value

    def fn_fetch_policy_table():
        return _policy_table

    return fn_get_policy_state_value, fn_set_policy_state_value, fn_fetch_policy_table