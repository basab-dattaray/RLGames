
from ws.RLUtils.setup.agent_dispatcher import agent_dispatcher


def fn_execute():
    agent_mgr = agent_dispatcher(__file__)
    agent_mgr. \
    agent_mgt(__file__). \
        fn_change_args({
            'NUM_MC_SIMULATIONS': 50,
            'NUM_TEST_GAMES': 12
        }). \
        fn_show_args(). \
        fn_test_against_greedy(). \
        fn_test_against_random()

    return agent_mgr.APP_INFO.ERROR_MESSAGE_


if __name__ == "__main__":
    print(fn_execute())