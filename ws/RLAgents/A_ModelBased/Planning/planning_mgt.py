from collections import namedtuple


def planning_mgt(env, discount_factor= 0.9):
    LOW_NUMBER = -9999999999
    _env_config = env.fn_get_config()

    def fn_get_actions_given_state(state):
        if env.fn_is_goal_reached(state):
            return None
        actions = env.Policy.fn_get_policy_state_value(state)
        return actions

    def fn_update_state_weighted_values_given_policy():

        for state in env.fn_get_all_states():
            env.fn_set_active_state(state)

            if env.fn_is_goal_reached(state):
                env.StateValues.fn_set_state_value(state, 0.0)
                continue

            average_value = 0.0
            action_value_list = fn_get_action_values()
            for action, value in action_value_list:
                average_value += fn_get_actions_given_state(state)[action] * value

            env.StateValues.fn_set_state_value(state, average_value)
        state_values = env.StateValues.fn_get_all_state_values()
        return state_values, None

    def fn_update_state_max_of_values_given_policy():

        for state in env.fn_get_all_states():

            if env.fn_is_goal_reached(state):
                env.StateValues.fn_set_state_value(state, 0)
                continue

            env.fn_set_active_state(state)

            action_value_list = fn_get_action_values()
            value_list = map(lambda av: av[1], action_value_list)

            max_value = round(max(value_list), 2)

            env.StateValues.fn_set_state_value(state, max_value)
        state_values = env.StateValues.fn_get_all_state_values()
        return state_values, None

    def fn_get_action_values():
        value_list = []
        for action in env.fn_value_table_possible_actions():
            next_state, reward, _, _ = env.fn_take_step(action, planning_mode=True)
            next_value = env.StateValues.fn_get_state_value(next_state)
            value_list.append((action, (reward + discount_factor * next_value)))
        return value_list

    def fn_run_policy_improvement():
        for state in env.fn_get_all_states():
            if env.fn_is_goal_reached(state):
                continue

            env.fn_set_active_state(state)

            value = LOW_NUMBER
            max_index = []

            for action in range(env.fn_get_action_size()):
                next_state, reward, _, _ = env.fn_take_step(action, planning_mode = True)
                next_value = env.StateValues.fn_get_state_value(next_state)
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

            env.Policy.fn_set_policy_state_value(state, result)
        policy_table = env.Policy.fn_fetch_policy_table()
        return None, policy_table

    def fn_repeat_policy_improvement_and_evaluation(apply_policy):
        while True:
            _, policy_table = fn_run_policy_improvement()

            state_values, policy = apply_policy()

            if not env.StateValues.fn_has_any_state_changed():
                break
        value_table = env.StateValues.fn_get_all_state_values()
        return value_table, policy_table

    def fn_policy_iterator():
        return fn_repeat_policy_improvement_and_evaluation(fn_update_state_weighted_values_given_policy)

    def fn_value_iterator():
        return fn_repeat_policy_improvement_and_evaluation(fn_update_state_max_of_values_given_policy)

    ret_obj = namedtuple('_', [
        'fn_policy_iterator',
        'fn_value_iterator',
        'fn_get_actions_given_state',
        'fn_run_policy_improvement',
        'fn_update_state_weighted_values_given_policy',
        'fn_update_state_max_of_values_given_policy',
    ])

    ret_obj.fn_policy_iterator = fn_policy_iterator
    ret_obj.fn_value_iterator = fn_value_iterator
    ret_obj.fn_get_actions_given_state = fn_get_actions_given_state
    ret_obj.fn_run_policy_improvement = fn_run_policy_improvement
    ret_obj.fn_update_state_weighted_values_given_policy = fn_update_state_weighted_values_given_policy
    ret_obj.fn_update_state_max_of_values_given_policy = fn_update_state_max_of_values_given_policy
    return ret_obj