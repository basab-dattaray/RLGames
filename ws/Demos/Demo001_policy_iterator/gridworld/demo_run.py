from ws.RLInterfaces.PARAM_KEY_NAMES import STRATEGY
from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.setup.preparation_mgt import preparation_mgt

if __name__ == "__main__":
    app_info, env = preparation_mgt(__file__)

    subpackage_name = 'ws.RLAgents.{}'.format(app_info[STRATEGY])
    agent_mgt = load_function(function_name="agent_mgt", module_tag="agent_mgt", subpackage_tag=subpackage_name)
    fnInit = agent_mgt(app_info, env)
    fnInit()
