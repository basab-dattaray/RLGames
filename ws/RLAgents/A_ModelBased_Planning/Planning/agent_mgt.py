from collections import OrderedDict, namedtuple

from ws.RLAgents.A_ModelBased_Planning.Planning.impl_mgt import impl_mgt
from ws.RLUtils.monitoring.tracing.tracer import tracer


def agent_mgt(app_info, common_functions):
    # app_info = startup_mgt(file_path, __file__)

    fn_move_per_policy, fn_apply_policy_iteration, fn_apply_value_iteration = impl_mgt(app_info)
    strategy = app_info.STRATEGY
    right_dot_index = strategy.rfind('.')
    iterator_name = strategy[right_dot_index + 1:]

    fn_apply = None
    if iterator_name == 'policy_iterator':
        fn_apply = fn_apply_policy_iteration
    if iterator_name == 'value_iterator':
        fn_apply = fn_apply_value_iteration

    # def fn_init():
    #     actions = OrderedDict()
    #     actions["plan"] = fn_apply
    #     actions["move"] = fn_move_per_policy
    #
    #     app_info.ENV.display_mgr.fn_setup_ui(actions)
    #
    #     app_info.ENV.display_mgr.fn_run_ui()
    #     # app_info.ENV.display_mgr.fn_close()
    #
    #     return agent_mgr

    def fn_setup_env():
        actions = OrderedDict()
        actions["plan"] = fn_apply
        actions["move"] = fn_move_per_policy

        app_info.ENV.display_mgr.fn_setup_ui(actions)
        return agent_mgr

    def fn_run_env():
        if 'TEST_MODE' in app_info:
            if app_info.TEST_MODE:
                app_info.ENV.display_mgr.fn_set_test_mode()
        app_info.ENV.display_mgr.fn_run_ui()
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