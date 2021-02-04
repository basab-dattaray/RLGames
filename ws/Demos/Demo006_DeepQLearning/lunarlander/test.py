from ws.RLAgents.C_ValueBase_WithFunctionApproximation.OffPolicy.dqn.agent_mgt import agent_mgt


def fn_exec_test():
    agent_mgr = agent_mgt(__file__). \
        fn_change_args({
            'TEST_MODE': 1,
        }). \
        fn_train()
    return agent_mgr.APP_INFO.ERROR_MESSAGE


if __name__ == "__main__":
    fn_exec_test()




