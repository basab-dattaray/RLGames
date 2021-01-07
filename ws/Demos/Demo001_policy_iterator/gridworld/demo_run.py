from ws.RLAgents.algo_lib.logic.support.agent_config_mgt import agent_config_mgt
from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.setup.preparation_mgt import preparation_mgt

if __name__ == "__main__":
    app_info, env = preparation_mgt(__file__)
    agent_config_mgt(app_info)
    subpackage_name = app_info['AGENTS_DOT_PATH'] + '.{}'.format(app_info['STRATEGY'])
    agent_mgt = load_function(function_name="agent_mgt", module_tag="agent_mgt", subpackage_tag=subpackage_name)
    agent = agent_mgt(app_info, env)
    agent.fn_init()
