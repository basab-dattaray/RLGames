from collections import OrderedDict, namedtuple

from ws.RLAgents.CAT1_model_based.planning.policy_iterator.impl_mgt import impl_mgt
from ws.RLUtils.setup.startup_mgt import startup_mgt


def agent_mgt(file_path):
    app_info = startup_mgt(file_path)

    fn_bind_fn_display_actions, fn_move_per_policy, fn_apply_policy_iteration, fn_apply_value_iteration = impl_mgt(app_info)

    def fn_init():
        actions = OrderedDict()
        actions["plan"] = fn_apply_value_iteration
        actions["move"] = fn_move_per_policy

        fn_bind_fn_display_actions(actions)
        return


    agent_mgr = namedtuple('_',
                                [
                                    'fn_init'
                                ]
                           )
    agent_mgr.fn_init = fn_init

    return agent_mgr
