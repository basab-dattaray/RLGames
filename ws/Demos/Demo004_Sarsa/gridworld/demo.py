from ws.RLAgents.A_ModelBased_Planning.Bootstrapping.OnPolicy_Bootstrapping.sarsa.agent_mgt import agent_mgt

if __name__ == "__main__":

    agent = agent_mgt(__file__). \
        fn_init()