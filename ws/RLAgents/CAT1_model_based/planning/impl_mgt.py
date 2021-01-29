import numpy as np

from ws.RLUtils.algo_lib.planning.planning_mgt import planning_mgt
from ws.RLEnvironments.gridworld.grid_board.display_mgt import display_mgt


def impl_mgt(app_info):

    _planning_mgr = planning_mgt(app_info.ENV, app_info.DISCOUNT_FACTOR)
    _display_mgr = display_mgt(app_info.ENV)

    def fn_bind_fn_display_actions(acton_dictionary):
        _display_mgr.fn_init(acton_dictionary)

    def fnNextGetAction(state):
        actions = _planning_mgr.fn_get_actions_given_state(state)
        best_action = np.random.choice(len(actions), p=actions)
        return best_action

    def fn_move_per_policy():
        start_state = _display_mgr.fn_get_start_state()
        state = _display_mgr.fn_run_next_move(app_info.ENV, start_state, fnNextGetAction)

        while state is not None:
            _display_mgr.fn_move_cursor(start_state, state)
            if _display_mgr.fn_is_target_state_reached(state):
                break
            start_state = state
            state = _display_mgr.fn_run_next_move(app_info.ENV, start_state, fnNextGetAction)
        _display_mgr.fn_move_cursor(state)

    def fn_apply_policy_iteration():
        value_table, policy_table = _planning_mgr.fnPolicyIterater()
        _display_mgr.fn_show_state_values(value_table)
        _display_mgr.fn_show_policy_arrows(policy_table)

    def fn_apply_value_iteration():
        value_table, _policy_table = _planning_mgr.fnValueIterater()
        _display_mgr.fn_show_state_values(value_table)
        _display_mgr.fn_show_policy_arrows(_policy_table)

    return fn_bind_fn_display_actions, fn_move_per_policy, fn_apply_policy_iteration, fn_apply_value_iteration
