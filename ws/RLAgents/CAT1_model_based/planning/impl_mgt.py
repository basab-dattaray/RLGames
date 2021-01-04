import random
from ws.RLAgents.CAT1_model_based.planning.planning_mgt import planning_mgt
from ws.RLEnvironments.gridworld.grid_board.display_mgt import display_mgt


def impl_mgt(env, app_info):
    _env = env

    fnPolicyIterater, fnValueIterater, fnGetValueFromPolicy = planning_mgt(env, app_info)
    _fn_display_controller = display_mgt(app_info)

    def fn_bind_fn_display_actions(acton_dictionary):

        _fn_display_controller.fn_init(acton_dictionary)

    def fnNextGetAction(state):
        random_pick = random.randrange(100) / 100

        policy_value_for_state = fnGetValueFromPolicy(state)

        if policy_value_for_state is None:
            return -1

        policy_sum = 0.0
        for index, value in enumerate(policy_value_for_state):
            policy_sum += value
            if random_pick < policy_sum:
                return index
        return -1

    def fn_move_per_policy():
        start_state = _fn_display_controller.fn_get_start_state()
        state = _fn_display_controller.fn_run_next_move(start_state, fnNextGetAction)

        while state is not None:
            _fn_display_controller.fn_move_cursor(start_state, state)
            if _fn_display_controller.fn_is_target_state_reached(state):
                break
            start_state = state
            state = _fn_display_controller.fn_run_next_move(start_state, fnNextGetAction)
        _fn_display_controller.fn_move_cursor(state)

    def fn_apply_policy_iteration():
        value_table, policy_table = fnValueIterater()
        _fn_display_controller.fn_show_state_values(value_table)
        _fn_display_controller.fn_show_policy_arrows(policy_table)

    def fn_apply_value_iteration():
        value_table, _policy_table = fnValueIterater()
        _fn_display_controller.fn_show_state_values(value_table)
        _fn_display_controller.fn_show_policy_arrows(_policy_table)

    return fn_bind_fn_display_actions, fn_move_per_policy, fn_apply_policy_iteration, fn_apply_value_iteration
