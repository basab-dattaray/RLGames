from ws.RLAgents.CAT1_model_based.planning.policy_iterator.agent_mgt import agent_mgt
from ws.RLAgents.algo_lib.logic.support.agent_config_mgt import agent_config_mgt
# from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.setup.preparation_mgt import preparation_mgt

if __name__ == "__main__":
    # app_info, env = preparation_mgt(__file__)
    # agent_config_mgt(app_info)
    agent = agent_mgt(__file__)
    agent.fn_init()
