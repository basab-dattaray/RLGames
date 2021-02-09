from collections import namedtuple

import numpy as np

from ws.RLAgents.A_ModelBased.Planning.planning_mgt import planning_mgt

def impl_mgt(app_info):

    _planner = planning_mgt(app_info.ENV, app_info.DISCOUNT_FACTOR)
    Display = app_info.ENV.Display

    def fn_next_get_action(state):
        actions = _planner.fn_get_actions_given_state(state)
        best_action = np.random.choice(len(actions), p=actions)
        return best_action

    def fn_move_per_policy():
        start_state = Display.fn_get_start_state()
        state = Display.fn_run_next_move(app_info.ENV, start_state, fn_next_get_action)

        while state is not None:
            Display.fn_move_cursor(start_state, state)
            if Display.fn_is_target_state_reached(state):
                break
            start_state = state
            state = Display.fn_run_next_move(app_info.ENV, start_state, fn_next_get_action)
        Display.fn_move_cursor(state)

    # def fn_apply_policy_iteration():
    #     value_table, policy_table = _planner.fn_policy_iterator()
    #     Display.fn_show_state_values(value_table)
    #     Display.fn_show_policy_arrows(policy_table)
    #
    # def fn_apply_value_iteration():
    #     value_table, policy_table = _planner.fn_value_iterator()
    #     Display.fn_show_state_values(value_table)
    #     Display.fn_show_policy_arrows(policy_table)

    def fn_display_therafter(fn):
        def augmented_fn():
            value_table, policy_table =fn()
            Display.fn_show_state_values(value_table)
            Display.fn_show_policy_arrows(policy_table)
        return augmented_fn

    def fn_apply_reset():
        app_info.ENV.fn_reset_env()
        StateValues, Policy = app_info.ENV.fn_get_internal_info()
        Display.fn_show_state_values(StateValues.fn_get_all_state_values(), show= False)
        Display.fn_show_policy_arrows(Policy.fn_fetch_policy_table(), show= False)



    ret_obj = namedtuple('_',
                         [
                             'Planner',
                             'fn_display_therafter',
                             'fn_move_per_policy',
                             # 'fn_apply_policy_iteration',
                             # 'fn_apply_value_iteration',
                             'fn_apply_reset',
                         ])
    ret_obj.Planner = _planner
    ret_obj.fn_display_therafter = fn_display_therafter
    ret_obj.fn_move_per_policy = fn_move_per_policy
    # ret_obj.fn_apply_policy_iteration = fn_apply_policy_iteration
    # ret_obj.fn_apply_value_iteration = fn_apply_value_iteration
    ret_obj.fn_apply_reset = fn_apply_reset

    return ret_obj
