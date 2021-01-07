
import coloredlogs

# from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.full_with_recursive_mcts.ARGS import args
from ws.RLAgents.CAT4_self_play.alpha_zero.misc.agent_mgt import agent_mgt

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

if __name__ == "__main__":
    agent_mgt(args= None, file_path= __file__). \
        fn_change_args({
        }). \
        fn_train().\
        fn_change_args({
            'num_of_mc_simulations': 50,
        }). \
        fn_test_against_greedy(). \
        fn_test_against_random(). \
        fn_measure_time_elapsed() . \
        fn_archive_log_file()