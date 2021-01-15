# from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.app_info import app_info
from ws.RLAgents.CAT4_self_play.alpha_zero.agent_mgt import agent_mgt

if __name__ == "__main__":
    agent_mgt( __file__). \
        fn_change_args({
            'NUM_MC_SIMULATIONS': 50,
            'NUM_TEST_GAMES': 12,
        }). \
        fn_show_args(). \
        fn_test_against_greedy()