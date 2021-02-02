# import coloredlogs

from ws.RLAgents.E_SelfPlay.DeepLearning_SelfPlay.agent_mgt import agent_mgt

# coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

if __name__ == "__main__":
    agent_mgt(file_path= __file__). \
        fn_change_args({
        }). \
        fn_train().\
        fn_change_args({
            'NUM_MC_SIMULATIONS': 50,
        }). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_archive_log_file()