import coloredlogs

from ws.RLAgents.CAT4_self_play.alpha_zero.misc.agent_mgt import agent_mgt

coloredlogs.install(level='INFO')

if __name__ == "__main__":
    agent_mgt(file_path= __file__). \
        \
        fn_reset(). \
        fn_change_args({
            'do_load_model': False,
            'mcts_ucb_use_action_prob_for_exploration': False,
            'mcts_ucb_use_log_in_numerator': False,
        }). \
        fn_train(). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_measure_time_elapsed(). \
        \
        fn_reset(). \
        fn_change_args({
            'do_load_model': False,
            'mcts_ucb_use_action_prob_for_exploration': True,
            'mcts_ucb_use_log_in_numerator': True,
        }). \
        fn_train(). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_measure_time_elapsed(). \
        fn_archive_log_file()
