
from ws.RLUtils.setup.agent_dispatcher import agent_dispatcher


def fn_execute():
    agent_mgr = agent_dispatcher(__file__)
    agent_mgr. \
        fn_change_args({
            'DO_LOAD_MODEL': True,
            'NUM_MC_SIMULATIONS': 50,
            'NUM_EPOCHS': 5,
            'UCB_USE_POLICY_FOR_EXPLORATION': True,
        }). \
        fn_train(). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_archive_log_file()
    return agent_mgr.APP_INFO.ERROR_MESSAGE_


if __name__ == "__main__":
    print(fn_execute())