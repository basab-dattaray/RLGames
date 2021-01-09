from collections import OrderedDict, namedtuple

from .impl_mgt import impl_mgt
from ws.RLAgents.algo_lib.logic.support.agent_config_mgt import agent_config_mgt
from ws.RLUtils.setup.startup_mgt import startup_mgt


def agent_mgt(file_path):
    app_info = startup_mgt(file_path)
    agent_config_mgt(app_info)
    fn_bind_fn_display_actions, fnRun = impl_mgt(app_info)
    def fn_init():
        actions = OrderedDict()
        actions["run"] = fnRun

        fn_bind_fn_display_actions(actions)
        return

    agent_mgr = namedtuple('_',
                                [
                                    'fn_init'
                                ]
                           )
    agent_mgr.fn_init = fn_init

    return agent_mgr

