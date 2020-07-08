from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.ARGS import args
from ws.RLAgents.self_play.alpha_zero.misc.Agent import Agent

if __name__ == "__main__":
    Agent.fn_init(args, __file__). \
        fn_change_args({
            'numMCTSSims': 50,
        }). \
        fn_show_args(). \
        fn_test_against_human()