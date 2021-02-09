from collections import OrderedDict, namedtuple

from ws.RLAgents.A_ModelBased.Planning.impl_mgt import impl_mgt


def agent_mgt(app_info, common_functions):

    implementation = impl_mgt(app_info)
    def fn_setup_env():
        actions = OrderedDict()
        actions["reset"] = implementation.fn_apply_reset
        actions["improve policy"] = implementation.fn_apply_reset
        actions["apply policy"] = implementation.fn_apply_reset
        actions["plan"] = implementation.fn_display_therafter(implementation.Planner.fn_value_iterator)
        actions["move"] = implementation.fn_move_per_policy

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