# from ws.RLAgents.A_ModelBased_Planning.Bootstrapping.OffPolicy.qlearn.agent_mgt import agent_mgt
# from ws.RLAgents.B_ValueBased.Bootstrapping.OffPolicy.qlearn.agent_mgt import agent_mgt
from ws.RLUtils.setup.agent_dispatcher import agent_dispatcher


def fn_exec_test():
    agent_mgr = agent_dispatcher(__file__)
    agent_mgr. \
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




