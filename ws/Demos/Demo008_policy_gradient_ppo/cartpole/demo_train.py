from ws.RLAgents.CAT3_policy_gradient_based.agent_mgt import agent_mgt
from ws.RLUtils.setup.preparation_mgt import preparation_mgt

if __name__ == '__main__':

    app_info, env = preparation_mgt(__file__)

    fn_run_train, fn_run_test = agent_mgt(app_info, env)
    fn_run_train()

