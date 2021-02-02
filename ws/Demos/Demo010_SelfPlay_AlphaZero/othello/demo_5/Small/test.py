from ws.RLAgents.Category4_SelfPlay.DeepLearning_SelfPlay.agent_mgt import agent_mgt


def fn_exec_test():
    agent_mgr = agent_mgt(__file__). \
        fn_change_args({
            'TEST_MODE_': 1,
            'NUM_TRAINING_ITERATIONS': 1,
            'NUM_TRAINING_EPISODES': 2,
            'DO_LOAD_MODEL': True,
            'PASSING_SCORE': 0.0,
            'NUM_GAMES_FOR_MODEL_COMPARISON': 4,
            'NUM_MC_SIMULATIONS': 5,
            'NUM_EPOCHS': 2,
            'NUM_TEST_GAMES': 2,
            'UCB_USE_POLICY_FOR_EXPLORATION': True,
        }). \
        fn_train(). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_archive_log_file()
    return agent_mgr.APP_INFO.ERROR_MESSAGE


if __name__ == "__main__":
    fn_exec_test()




