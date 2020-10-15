from ws.RLAgents.model_based.planning.impl_mgt import impl_mgt


def agent_mgt(app_info, env):
    def fnInit():
        fn_bind_display_actions, fn_move_per_policy, fn_apply_policy_iteration, fn_apply_value_iteration = impl_mgt(env, app_info)
        fn_bind_display_actions({"plan": fn_apply_policy_iteration, "move": fn_move_per_policy})

    return fnInit
