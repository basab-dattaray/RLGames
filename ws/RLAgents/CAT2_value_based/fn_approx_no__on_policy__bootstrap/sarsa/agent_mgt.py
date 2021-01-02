from .impl_mgt import impl_mgt


def agent_mgt(app_info, env):
    def fn_init():
        fn_bind_fn_display_actions, fnRun = impl_mgt(env, app_info)
        fn_bind_fn_display_actions({"run": fnRun})
        return fn_bind_fn_display_actions

    return fn_init
