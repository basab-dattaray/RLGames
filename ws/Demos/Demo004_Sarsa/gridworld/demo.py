from ws.RLAgents.B_ValueBased.Bootstrapping.OnPolicy.sarsa.agent_mgt import agent_mgt

if __name__ == "__main__":

    agent = agent_mgt(__file__). \
        fn_init()