from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.full_with_recursive_mcts.ARGS import args_out
from ws.RLAgents.self_play.alpha_zero.misc.agent_mgt import agent_mgt

if __name__ == "__main__":
    agent_mgt(args_out, __file__). \
        fn_change_args({
            'NUM_MC_SIMULATIONS': 50,
        }). \
        fn_show_args(). \
        fn_test_against_human()