import coloredlogs

from ws.RLAgents.CAT4_self_play.alpha_zero.misc.agent_mgt import agent_mgt

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

if __name__ == "__main__":
    agent_mgt(file_path= __file__). \
        fn_change_args({
            'do_load_model': True,
            'num_of_mc_simulations': 50,
            'epochs': 5,
            # 'mcts_ucb_use_action_prob_for_exploration': True,
        }). \
        fn_train(). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_measure_time_elapsed(). \
        fn_archive_log_file()
