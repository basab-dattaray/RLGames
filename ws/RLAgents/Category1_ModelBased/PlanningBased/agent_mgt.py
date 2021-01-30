from collections import OrderedDict, namedtuple

from ws.RLAgents.Category1_ModelBased.PlanningBased.impl_mgt import impl_mgt
from ws.RLUtils.setup.startup_mgt import startup_mgt


def agent_mgt(file_path):
    app_info = startup_mgt(file_path, __file__)

    fn_bind_fn_display_actions, fn_move_per_policy, fn_apply_policy_iteration, fn_apply_value_iteration = impl_mgt(app_info)
    strategy = app_info.STRATEGY
    right_dot_index = strategy.rfind('.')
    iterator_name =  strategy[right_dot_index + 1:]

    fn_apply = None
    if iterator_name == 'policy_iterator':
        fn_apply = fn_apply_policy_iteration
    if iterator_name == 'value_iterator':
        fn_apply = fn_apply_value_iteration



    def fn_init():
        actions = OrderedDict()
        actions["plan"] = fn_apply
        actions["move"] = fn_move_per_policy

        fn_bind_fn_display_actions(actions)
        return

    fn_init()

    return
