from collections import namedtuple


def planning_mgt(env, discount_factor= 0.9):
    LOW_NUMBER = -9999999999
    _env_config = env.fn_get_config()

    def fn_get_actions_given_state(state):
        if env.fn_is_goal_reached(state):
            return None
        actions = env.policy_repo_mgr.fn_get_policy_state_value(state)
        return actions

    def fn_run_policy():

        for state in env.fn_get_all_states():
            env.fn_set_active_state(state)

            if env.fn_is_goal_reached(state):
                env.values_repo_mgr.fn_set_state_value(state, 0.0)
                continue

            value = 0.0
            for action in env.fn_value_table_possible_actions():
                next_state, reward, _, _= env.fn_take_step(action, planning_mode = True)

                next_value = env.values_repo_mgr.fn_get_state_value(next_state)
                value = value + fn_get_actions_given_state(state)[action] * (reward + discount_factor * next_value)

            env.values_repo_mgr.fn_set_state_value(state, value)

    def fn_calc_values():

        for state in env.fn_get_all_states():

            if env.fn_is_goal_reached(state):
                env.values_repo_mgr.fn_set_state_value(state, 0)
                continue

            env.fn_set_active_state(state)

            value_list = []

            for action in env.fn_value_table_possible_actions():
                next_state, reward, _, _= env.fn_take_step(action, planning_mode = True)
                next_value = env.values_repo_mgr.fn_get_state_value(next_state)
                value_list.append((reward + discount_factor * next_value))

            env.values_repo_mgr.fn_set_state_value(state, round(max(value_list), 2))

    def fn_run_policy_improvement():
        for state in env.fn_get_all_states():
            if env.fn_is_goal_reached(state):
                continue

            env.fn_set_active_state(state)

            value = LOW_NUMBER
            max_index = []

            for action in range(env.fn_get_action_size()):
                next_state, reward, _, _ = env.fn_take_step(action, planning_mode = True)
                next_value = env.values_repo_mgr.fn_get_state_value(next_state)
                total_reward = reward + discount_factor * next_value

                if total_reward == value: # there can be multiple maximums
                    max_index.append(action)
                elif total_reward > value: # start new maximum maximums
                    value = total_reward
                    max_index.clear()
                    max_index.append(action)

            result = [0] * env.fn_get_action_size()

            prob = 1 / len(max_index)
            for index in max_index:
                result[index] = prob

            env.policy_repo_mgr.fn_set_policy_state_value(state, result)
        policy_table = env.policy_repo_mgr.fn_fetch_policy_table()
        return policy_table

    def repeatEvalAndImprove(evalFunction):
        while True:
            policy_table = fn_run_policy_improvement()

            evalFunction()

            if not env.values_repo_mgr.fn_has_any_state_changed():
                break
        value_table = env.values_repo_mgr.fn_get_all_state_values()
        return value_table, policy_table

    def fn_policy_iterator():
        return repeatEvalAndImprove(fn_run_policy)

    def fn_value_iterator():
        return repeatEvalAndImprove(fn_calc_values)

    ret_obj = namedtuple('_', [
        'fn_policy_iterator',
        'fn_value_iterator',
        'fn_get_actions_given_state',
    ])

    ret_obj.fn_policy_iterator = fn_policy_iterator
    ret_obj.fn_value_iterator = fn_value_iterator
    ret_obj.fn_get_actions_given_state = fn_get_actions_given_state
    return ret_obj