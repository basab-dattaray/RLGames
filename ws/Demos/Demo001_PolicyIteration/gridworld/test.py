from ws.RLAgents.A_ModelBased_Planning.Planning.agent_mgt_wrapper import agent_mgt_wrapper


def fn_exec_test():
    agent_mgr = agent_mgt_wrapper(__file__)
    agent_mgr. \
        fn_set_test_mode(). \
        fn_init()
    return agent_mgr.APP_INFO.ERROR_MESSAGE


if __name__ == "__main__":
    fn_exec_test()



