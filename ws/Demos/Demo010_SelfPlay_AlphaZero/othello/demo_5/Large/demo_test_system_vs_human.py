from ws.RLAgents.E_SelfPlay.DeepLearning_SelfPlay.agent_mgt import agent_mgt

if __name__ == "__main__":
    agent_mgt(__file__). \
        fn_change_args({
            'NUM_MC_SIMULATIONS': 50,
        }). \
        fn_show_args(). \
        fn_test_against_human()