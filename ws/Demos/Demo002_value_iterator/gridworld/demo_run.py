from ws.RLInterfaces.PARAM_KEY_NAMES import STRATEGY
from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.setup.preparation_mgt import preparation_mgr

if __name__ == "__main__":
    app_info, env = preparation_mgr(__file__)

    subpackage_name = 'ws.RLAgents.{}'.format(app_info[STRATEGY])
    agent_mgr = load_function(function_name="agent_mgr", module_tag="agent_mgt", subpackage_tag=subpackage_name)
    fnInit = agent_mgr(app_info, env)
    fnInit()
