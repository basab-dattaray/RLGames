from ws.RLAgents.Category1_ModelBased.BasedOnPlanning.agent_mgt import agent_mgt


def fn_exec_test():
    agent_mgr = agent_mgt(__file__). \
        fn_set_test_mode(). \
        fn_init()
    return agent_mgr.APP_INFO.ERROR_MESSAGE


if __name__ == "__main__":
    fn_exec_test()

