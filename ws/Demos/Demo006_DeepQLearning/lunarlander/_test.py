from ws.RLAgents.Category2_ValueBased.DeepLearning_OffPolicy_Bootstrapping.dqn.agent_mgt import agent_mgt


def fn_exec_test():
    agent_mgt(__file__). \
        fn_set_test_mode(). \
        fn_init()


if __name__ == "__main__":
    fn_exec_test()




