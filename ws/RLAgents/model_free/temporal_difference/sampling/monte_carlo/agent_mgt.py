from .impl_mgt import impl_mgr


def agent_mgr(app_info, env):
    def fnInit():
        fn_bind_display_actions, fnRun = impl_mgr(env, app_info)
        fn_bind_display_actions({"run": fnRun})

    return fnInit
