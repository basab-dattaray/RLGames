from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.full.ARGS import args
from ws.RLAgents.self_play.alpha_zero.misc.agent_mgt import agent_mgt

if __name__ == "__main__":
    agent_mgt(args, __file__). \
        fn_change_args({
            'num_of_mc_simulations': 50,
            'num_of_test_games': 12
        }). \
        fn_show_args(). \
        fn_test_against_greedy(). \
        fn_test_against_random()