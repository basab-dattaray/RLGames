from collections import OrderedDict, namedtuple

from ws.RLAgents.A_ModelBased.Planning.impl_mgt import impl_mgt


def agent_mgt(app_info, common_functions):
    # app_info = startup_mgt(file_path, __file__)

    fn_move_per_policy, fn_apply_policy_iteration, fn_apply_value_iteration, fn_apply_reset = impl_mgt(app_info)

    def fn_setup_env():
        actions = OrderedDict()
        actions["reset"] = fn_apply_reset
        actions["plan"] = fn_apply_value_iteration
        actions["move"] = fn_move_per_policy

        app_info.ENV.Display.fn_setup_ui(actions)
        return agent_mgr

    def fn_run_env():
        if 'TEST_MODE' in app_info:
            if app_info.TEST_MODE:
                app_info.ENV.Display.fn_set_test_mode()
        app_info.ENV.Display.fn_run_ui()
        return agent_mgr


    agent_mgr = namedtuple('_',
                                [
                                    'fn_setup_env',
                                    'fn_run_env',
                                    'fn_change_args',
                                    'APP_INFO',
                                ]
                           )
    agent_mgr.fn_setup_env = fn_setup_env
    agent_mgr.fn_run_env = fn_run_env
    agent_mgr.fn_change_args = common_functions.fn_change_args
    agent_mgr.APP_INFO = app_info


    return agent_mgr