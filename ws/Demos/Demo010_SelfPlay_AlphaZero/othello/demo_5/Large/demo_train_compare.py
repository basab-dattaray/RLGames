import coloredlogs

from ws.RLAgents.Category4_SelfPlay.DeepLearning_SelfPlay.agent_mgt import agent_mgt

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


if __name__ == "__main__":
    agent_mgt(file_path= __file__). \
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
        fn_archive_log_file(). \
        \
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
