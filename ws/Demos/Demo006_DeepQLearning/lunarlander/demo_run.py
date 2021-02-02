from ws.RLAgents.C_ValueBase_WithFunctionApproximation.DeepLearning_OffPolicy_Bootstrapping.dqn.agent_mgt import agent_mgt

if __name__ == "__main__":
    agent_mgr = agent_mgt(__file__). \
        fn_train()


