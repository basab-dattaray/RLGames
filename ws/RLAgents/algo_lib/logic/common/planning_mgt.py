from ws.RLAgents.algo_lib.logic.common.policy_table_mgt import policy_table_mgt
from ws.RLAgents.algo_lib.logic.common.value_table_mgt import value_table_mgt

def planning_mgt(app_info):
    LOW_NUMBER = -9999999999
    _env = app_info.ENV

    _discount_factor = app_info["DISCOUNT_FACTOR"]

    fn_set_value_table_item, fn_get_value_table_item, fn_set_value_table, fn_get_value_table, \
    _ ,fn_value_table_reached_target, fn_has_table_changed = \
                                            value_table_mgt(app_info)

    fn_get_policy_state_value, fn_set_policy_state_value, fn_fetch_policy_table = policy_table_mgt(app_info)

    def fnGetValueFromPolicy(state):
        if fn_value_table_reached_target(state):
            return None
        val = fn_get_policy_state_value(state)
        return val

    def applyPolicyIteration():

        for state in _env.fn_get_all_states():

            _env.fn_update_state(state)


            if fn_value_table_reached_target(state):
                fn_set_value_table_item(state, 0.0)
                continue

            value = 0.0
            for action in _env.fn_value_table_possible_actions():
                next_state, reward, _, _= _env.fn_take_step(action, planning_mode = True)

                next_value = fn_get_value_table_item(next_state)
                value = value + fnGetValueFromPolicy(state)[action] * (reward + _discount_factor * next_value)

            fn_set_value_table_item(state, value)

    def applyValueIteration():
        all_states = _env.fn_get_all_states()

        for state in all_states:

            if fn_value_table_reached_target(state):
                fn_set_value_table_item(state, 0)
                continue

            _env.fn_update_state(state)

            value_list = []

            for action in _env.fn_value_table_possible_actions():
                next_state, reward, _, _= _env.fn_take_step(action, planning_mode = True)
                next_value = fn_get_value_table_item(next_state)
                value_list.append((reward + _discount_factor * next_value))

            fn_set_value_table_item(state, round(max(value_list), 2))

    def improvePolicy():
        for state in _env.fn_get_all_states():
            if fn_value_table_reached_target(state):
                continue

            _env.fn_update_state(state)

            value = LOW_NUMBER
            max_index = []

            for index, action in enumerate(_env.fn_value_table_possible_actions()):
                next_state, reward, _, _ = _env.fn_take_step(action, planning_mode = True)
                next_value = fn_get_value_table_item(next_state)
                total_reward = reward + _discount_factor * next_value

                if total_reward == value: # there can be multiple maximums
                    max_index.append(index)
                elif total_reward > value: # start new maximum maximums
                    value = total_reward
                    max_index.clear()
                    max_index.append(index)

            prob = 1 / len(max_index)

            result = app_info['INITIAL_ACTION_VALUES'] .copy()
            for index in max_index:
                result[index] = prob
            fn_set_policy_state_value(state, result)
        policy_table = fn_fetch_policy_table()
        return policy_table

    def repeatEvalAndImprove(evalFunction):
        while True:
            policy_table = improvePolicy()

            evalFunction()

            if not fn_has_table_changed():
                break
        value_table = fn_get_value_table()
        return value_table, policy_table

    def fnPolicyIterater():
        return repeatEvalAndImprove(applyPolicyIteration)

    def fnValueIterater():
        return repeatEvalAndImprove(applyValueIteration)


    return fnPolicyIterater, fnValueIterater, fnGetValueFromPolicy