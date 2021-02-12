from collections import OrderedDict, namedtuple

from ws.RLAgents.A_ModelBased.Planning.impl_mgt import impl_mgt


def agent_mgt(app_info, common_functions):

    implementation = impl_mgt(app_info)
    def fn_setup_env():
        actions = OrderedDict()
        actions["reset"] = implementation.fn_display_therafter(implementation.Planner.fn_reset)
        # actions["reset"] = implementation.fn_display_therafter(implementation.fn_reset_planner)
        actions["improve policy"] = implementation.fn_display_therafter(
            implementation.Planner.fn_run_policy_improvement)
        actions["apply policy"] = implementation.fn_display_therafter(
            implementation.Planner.fn_update_state_value_by_choosing_best_action)
        actions["plan"] = implementation.fn_display_therafter(implementation.Planner.fn_value_iterator)
        actions["move"] = implementation.fn_move_per_policy

        implementation.fn_show_grid(actions)
        return agent_mgr

    # def fn_show_grid(actions):
    #     app_info.ENV.Display.fn_setup_ui(actions, app_info.ENV)
    #     Values, Policy = app_info.ENV.fn_get_internal_info()
    #     values = Values.fn_fetch_state_values()
    #     app_info.ENV.Display.fn_show_state_values(values)
    #     policy = Policy.fn_fetch_policy_table()
    #     app_info.ENV.Display.fn_show_policy_arrows(policy)

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