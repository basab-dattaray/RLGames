
from ws.RLUtils.setup.agent_dispatcher import agent_dispatcher


def fn_execute():
    agent_mgr = agent_dispatcher(__file__)
    agent_mgr. \
        fn_change_args({
            'TEST_MODE_': 1,
            'NUM_TRAINING_ITERATIONS': 1,
            'NUM_TRAINING_EPISODES': 2,
            'DO_LOAD_MODEL': True,
            'PASSING_SCORE': 0.0,
            'NUM_GAMES_FOR_MODEL_COMPARISON': 2,
            'NUM_MC_SIMULATIONS': 3,
            'NUM_EPOCHS': 2,
            'NUM_TEST_GAMES': 2,
            'UCB_USE_POLICY_FOR_EXPLORATION': True,
        }). \
        fn_train(). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_archive_log_file()
    return agent_mgr.APP_INFO.ERROR_MESSAGE_


if __name__ == "__main__":
    print(fn_execute())




