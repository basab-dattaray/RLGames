from collections import namedtuple

from ws.RLUtils.algo_lib.planning.policy_table_mgt import policy_table_mgt
from ws.RLUtils.algo_lib.planning.value_table_mgt import value_table_mgt

def planning_mgt(app_info):
    LOW_NUMBER = -9999999999
    _env = app_info.ENV

    _discount_factor = app_info.DISCOUNT_FACTOR

    fn_set_value_table_item, fn_get_value_table_item, fn_set_value_table, fn_get_value_table, \
    _ ,fn_value_table_reached_target, fn_has_table_changed = \
                                            value_table_mgt(app_info)

    fn_get_policy_state_value, fn_set_policy_state_value, fn_fetch_policy_table = policy_table_mgt(app_info)

    def fn_get_actions_given_state(state):
        if fn_value_table_reached_target(state):
            return None
        actions = fn_get_policy_state_value(state)
        return actions

    def fn_run_policy():

        for sites in _env.fn_get_all_sites():

            _env.fn_update_current_site(sites)


            if fn_value_table_reached_target(sites):
                fn_set_value_table_item(sites, 0.0)
                continue

            value = 0.0
            for action in _env.fn_value_table_possible_actions():
                next_state, reward, _, _= _env.fn_take_step(action, planning_mode = True)

                next_value = fn_get_value_table_item(next_state)
                value = value + fn_get_actions_given_state(sites)[action] * (reward + _discount_factor * next_value)

            fn_set_value_table_item(sites, value)

    def fn_calc_values():

        for site in _env.fn_get_all_sites():

            if fn_value_table_reached_target(site):
                fn_set_value_table_item(site, 0)
                continue

            _env.fn_update_current_site(site)

            value_list = []

            for action in _env.fn_value_table_possible_actions():
                next_state, reward, _, _= _env.fn_take_step(action, planning_mode = True)
                next_value = fn_get_value_table_item(next_state)
                value_list.append((reward + _discount_factor * next_value))

            fn_set_value_table_item(site, round(max(value_list), 2))

    def fn_run_policy_improvement():
        for site in _env.fn_get_all_sites():
            if fn_value_table_reached_target(site):
                continue

            _env.fn_update_current_site(site)

            value = LOW_NUMBER
            max_index = []

            for action in range(_env.fn_get_action_size()):
                next_state, reward, _, _ = _env.fn_take_step(action, planning_mode = True)
                next_value = fn_get_value_table_item(next_state)
                total_reward = reward + _discount_factor * next_value

                if total_reward == value: # there can be multiple maximums
                    max_index.append(action)
                elif total_reward > value: # start new maximum maximums
                    value = total_reward
                    max_index.clear()
                    max_index.append(action)

            prob = 1 / len(max_index)

            result = [0] * _env.fn_get_action_size()

            for index in max_index:
                result[index] = prob
            fn_set_policy_state_value(site, result)
        policy_table = fn_fetch_policy_table()
        return policy_table

    def repeatEvalAndImprove(evalFunction):
        while True:
            policy_table = fn_run_policy_improvement()

            evalFunction()

            if not fn_has_table_changed():
                break
        value_table = fn_get_value_table()
        return value_table, policy_table

    def fnPolicyIterater():
        return repeatEvalAndImprove(fn_run_policy)

    def fnValueIterater():
        return repeatEvalAndImprove(fn_calc_values)

    ret_obj = namedtuple('_', [
        'fnPolicyIterater',
        'fnValueIterater',
        'fn_get_actions_given_state',
    ])

    ret_obj.fnPolicyIterater = fnPolicyIterater
    ret_obj.fnValueIterater = fnValueIterater
    ret_obj.fn_get_actions_given_state = fn_get_actions_given_state
    return ret_obj