from collections import namedtuple

from ws.RLUtils.algo_lib.planning.policy_table_mgt import policy_table_mgt
from ws.RLUtils.algo_lib.planning.value_table_mgt import value_table_mgt

def planning_mgt(env, discount_factor= 0.9):
    LOW_NUMBER = -9999999999
    _env_config = env.fn_get_config()

    fn_set_state_statepart_value, fn_get_state_statepart_value, fn_set_all_state_statepart_values, fn_get_all_state_statepart_values, \
    _ ,fn_has_any_state_statepart_changed = value_table_mgt(
        env,
    )

    policy_repo = policy_table_mgt(env)

    def fn_get_actions_given_state(state):
        if env.fn_is_goal_reached(state):
            return None
        actions = policy_repo.fn_get_policy_state_value(state)
        return actions

    def fn_run_policy():

        for statepart in env.fn_get_all_stateparts():
            env.fn_set_active_statepart(statepart)

            if env.fn_is_goal_reached(statepart):
                fn_set_state_statepart_value(statepart, 0.0)
                continue

            value = 0.0
            for action in env.fn_value_table_possible_actions():
                next_state, reward, _, _= env.fn_take_step(action, planning_mode = True)

                next_value = fn_get_state_statepart_value(next_state)
                value = value + fn_get_actions_given_state(statepart)[action] * (reward + discount_factor * next_value)

            fn_set_state_statepart_value(statepart, value)

    def fn_calc_values():

        for statepart in env.fn_get_all_stateparts():

            if env.fn_is_goal_reached(statepart):
                fn_set_state_statepart_value(statepart, 0)
                continue

            env.fn_set_active_statepart(statepart)

            value_list = []

            for action in env.fn_value_table_possible_actions():
                next_state, reward, _, _= env.fn_take_step(action, planning_mode = True)
                next_value = fn_get_state_statepart_value(next_state)
                value_list.append((reward + discount_factor * next_value))

            fn_set_state_statepart_value(statepart, round(max(value_list), 2))

    def fn_run_policy_improvement():
        for statepart in env.fn_get_all_stateparts():
            if env.fn_is_goal_reached(statepart):
                continue

            env.fn_set_active_statepart(statepart)

            value = LOW_NUMBER
            max_index = []

            for action in range(env.fn_get_action_size()):
                next_state, reward, _, _ = env.fn_take_step(action, planning_mode = True)
                next_value = fn_get_state_statepart_value(next_state)
                total_reward = reward + discount_factor * next_value

                if total_reward == value: # there can be multiple maximums
                    max_index.append(action)
                elif total_reward > value: # start new maximum maximums
                    value = total_reward
                    max_index.clear()
                    max_index.append(action)

            prob = 1 / len(max_index)

            result = [0] * env.fn_get_action_size()

            for index in max_index:
                result[index] = prob
            policy_repo.fn_set_policy_state_value(statepart, result)
        policy_table = policy_repo.fn_fetch_policy_table()
        return policy_table

    def repeatEvalAndImprove(evalFunction):
        while True:
            policy_table = fn_run_policy_improvement()

            evalFunction()

            if not fn_has_any_state_statepart_changed():
                break
        value_table = fn_get_all_state_statepart_values()
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