from collections import OrderedDict, namedtuple

from ws.RLUtils.monitoring.tracing.tracer import tracer
from .impl_mgt import impl_mgt

def agent_mgt(app_info, common_functions):
    fn_run = impl_mgt(app_info)

    def fn_setup_env():
        actions = OrderedDict()
        actions["run"] = fn_run

        app_info.ENV.display_mgr.fn_setup_ui(actions)
        return agent_mgr

    def fn_run_env():
        if 'TEST_MODE' in app_info:
            if app_info.TEST_MODE:
                app_info.ENV.display_mgr.fn_set_test_mode()
        app_info.ENV.display_mgr.fn_run_ui()
        return agent_mgr

    @tracer(app_info, verboscity= 4)
    def fn_change_args(change_args):
        if change_args is not None:
            for k, v in change_args.items():
                app_info[k] = v
                app_info.trace_mgr.fn_write(f'  app_info[{k}] = {v}')
        agent_mgr.app_info = app_info
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
