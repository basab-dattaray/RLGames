from .impl_mgt import impl_mgt


def agent_mgt(app_info, env):
    def fnInit():
        fn_bind_display_actions, fnRun = impl_mgt(env, app_info)
        fn_bind_display_actions({"run": fnRun})

    return fnInit
