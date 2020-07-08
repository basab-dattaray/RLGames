from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.full.ARGS import args
from ws.RLAgents.self_play.alpha_zero.misc.Agent import Agent

if __name__ == "__main__":
    Agent.fn_init(args, __file__). \
        fn_change_args({
            'numMCTSSims': 50,
            'num_of_test_games': 500,
        }). \
        fn_show_args(). \
        fn_test_against_greedy()