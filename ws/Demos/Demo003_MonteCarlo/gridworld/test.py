# from ws.RLAgents.A_ModelBased_Planning.Sampling.OnPolicy_Sampling.monte_carlo.agent_mgt import agent_mgt
from ws.RLAgents.B_ValueBased.Sampling.OnPolicy_Sampling.monte_carlo.agent_mgt import agent_mgt


def fn_exec_test():
    agent_mgr = agent_mgt(__file__). \
        fn_set_test_mode(). \
        fn_init()
    return agent_mgr.APP_INFO.ERROR_MESSAGE

if __name__ == "__main__":
    fn_exec_test()


