import random

from ws.RLAgents.model_based.planning.planning_mgt import planning_mgt


# from ws.RLEnvironments.gridworld.logic.SETUP_INFO import ACTION_MOVE_STATE_RULES

def impl_mgt(env, app_info):
    _env = env

    _fn_display_controller, fnPolicyIterater, fnValueIterater, fnGetValueFromPolicy = planning_mgt(env, app_info)

    def fn_bind_fn_display_actions(acton_dictionary):

        _fn_display_controller.fnInit(acton_dictionary)

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
        start_state = _fn_display_controller.fnGetStartState()
        state = _fn_display_controller.fnExecNextMove(start_state, fnNextGetAction)

        while state is not None:
            _fn_display_controller.fnMoveCursor(start_state, state)
            if _fn_display_controller.fnIsTargetStateReached(state):
                break
            start_state = state
            state = _fn_display_controller.fnExecNextMove(start_state, fnNextGetAction)
        _fn_display_controller.fnMoveCursor(state)

    def fn_apply_policy_iteration():
        value_table, policy_table = fnValueIterater()
        _fn_display_controller.fnShowStateValues(value_table)
        _fn_display_controller.fnShowPolicyArrows(policy_table)

    def fn_apply_value_iteration():
        value_table, _policy_table = fnValueIterater()
        _fn_display_controller.fnShowStateValues(value_table)
        _fn_display_controller.fnShowPolicyArrows(_policy_table)

    return fn_bind_fn_display_actions, fn_move_per_policy, fn_apply_policy_iteration, fn_apply_value_iteration
