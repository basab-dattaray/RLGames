
from ws.RLUtils.setup.agent_dispatcher import agent_dispatcher

def fn_execute():
    agent_mgr = agent_dispatcher(__file__)
    agent_mgr. \
        fn_reset(). \
        fn_change_args({
            'DO_LOAD_MODEL': False,
            'UCB_USE_POLICY_FOR_EXPLORATION': False,
            'UCB_USE_LOG_IN_NUMERATOR': False,
        }). \
        fn_train(). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_measure_time_elapsed(). \
        \
        fn_reset(). \
        fn_change_args({
            'DO_LOAD_MODEL': False,
            'UCB_USE_POLICY_FOR_EXPLORATION': True,
            'UCB_USE_LOG_IN_NUMERATOR': True,
        }). \
        fn_train(). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_measure_time_elapsed(). \
        fn_archive_log_file()
    return agent_mgr.APP_INFO.ERROR_MESSAGE_


if __name__ == "__main__":
    print(fn_execute())