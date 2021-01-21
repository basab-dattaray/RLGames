def policy_iter_mgt(planning_mgr, fn_display_controller):
    def fn_apply_policy_iteration():
        value_table, policy_table = planning_mgr.fnPolicyIterater()
        fn_display_controller.fn_show_state_values(value_table)
        fn_display_controller.fn_show_policy_arrows(policy_table)

    return fn_apply_policy_iteration