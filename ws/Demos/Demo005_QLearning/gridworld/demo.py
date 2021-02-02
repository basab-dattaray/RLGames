from ws.RLAgents.Category2_ValueBased.OffPolicy_Bootstrapping.qlearn.agent_mgt import agent_mgt

def fn_exec_test():
    agent_mgr = agent_mgt(__file__). \
        fn_setup_env(). \
        fn_run_env()
    return agent_mgr.APP_INFO.ERROR_MESSAGE

if __name__ == "__main__":
    fn_exec_test()