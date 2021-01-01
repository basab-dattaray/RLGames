from collections import OrderedDict

from ws.RLAgents.model_based.planning.impl_mgt import impl_mgt


def agent_mgt(app_info, env):
    def fn_init():
        fn_bind_fn_display_actions, fn_move_per_policy, fn_apply_policy_iteration, fn_apply_value_iteration = impl_mgt(env, app_info)
        actions = OrderedDict()
        actions["plan"] = fn_apply_value_iteration
        actions["move"] = fn_move_per_policy

        fn_bind_fn_display_actions(actions) #{"plan": fn_apply_value_iteration, "move": fn_move_per_policy})

    return fn_init
