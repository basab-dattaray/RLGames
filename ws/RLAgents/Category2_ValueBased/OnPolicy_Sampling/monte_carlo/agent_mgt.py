from collections import OrderedDict, namedtuple

from ws.RLUtils.monitoring.tracing.tracer import tracer
from .impl_mgt import impl_mgt

from ws.RLUtils.setup.startup_mgt import startup_mgt


def agent_mgt(file_path):
    app_info = startup_mgt(file_path, __file__)

    fn_bind_fn_display_actions, fnRun = impl_mgt(app_info)
    def fn_init():
        actions = OrderedDict()
        actions["run"] = fnRun

        fn_bind_fn_display_actions(actions)
        return

    @tracer(app_info, verboscity= 4)
    def fn_set_test_mode():
        app_info.ENV.display_mgr.fn_set_test_mode()
        return agent_mgr


    agent_mgr = namedtuple('_',
                                [
                                    'fn_init',
                                    'fn_set_test_mode'
                                ]
                           )
    agent_mgr.fn_init = fn_init
    agent_mgr.fn_set_test_mode = fn_set_test_mode

    return agent_mgr

