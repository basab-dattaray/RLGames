def fn_apply_policy_iteration():
    value_table, policy_table = planning_mgr.fnPolicyIterater()
    _fn_display_controller.fn_show_state_values(value_table)
    _fn_display_controller.fn_show_policy_arrows(policy_table)