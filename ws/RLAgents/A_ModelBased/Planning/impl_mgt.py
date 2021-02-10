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

    def fn_display_therafter(fn):
        def augmented_fn():
            value_table, policy_table = fn()
            if value_table is not None:
                Display.fn_show_state_values(value_table)
            if policy_table is not None:
                Display.fn_show_policy_arrows(policy_table)
        return augmented_fn

    def fn_reset_planner():
        app_info.ENV.fn_reset_env()
        StateValues, Policy = app_info.ENV.fn_get_internal_info()

        values = StateValues.fn_fetch_state_values()
        Display.fn_show_state_values(values)

        policy = Policy.fn_fetch_policy_table()
        Display.fn_show_policy_arrows(policy)

        return values, policy



    ret_obj = namedtuple('_',
                         [
                             'Planner',
                             'fn_display_therafter',
                             'fn_move_per_policy',
                             'fn_reset_planner',
                         ])
    ret_obj.Planner = _planner
    ret_obj.fn_display_therafter = fn_display_therafter
    ret_obj.fn_move_per_policy = fn_move_per_policy
    ret_obj.fn_reset_planner = fn_reset_planner

    return ret_obj
