from ws.RLAgents.model_free.policy_gradient.agent_mgt import agent_mgr
from ws.RLUtils.setup.preparation_mgt import preparation_mgr

if __name__ == '__main__':

    app_info, env = preparation_mgr(__file__)

    fn_run_train, fn_run_test = agent_mgr(app_info, env)
    fn_run_test()

