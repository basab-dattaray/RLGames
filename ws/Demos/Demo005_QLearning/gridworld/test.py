# from ws.RLAgents.A_ModelBased_Planning.Bootstrapping.OffPolicy.qlearn.agent_mgt import agent_mgt
from ws.RLAgents.B_ValueBased.Bootstrapping.OffPolicy.qlearn.agent_mgt import agent_mgt


def fn_exec_test():
    agent_mgr = agent_mgt(__file__). \
        fn_change_args(
            {
                'TEST_MODE': True,
            }
        ). \
        fn_setup_env(). \
        fn_run_env()
    return agent_mgr.APP_INFO.ERROR_MESSAGE

if __name__ == "__main__":
    fn_exec_test()




