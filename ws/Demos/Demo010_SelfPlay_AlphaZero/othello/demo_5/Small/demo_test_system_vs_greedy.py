# from ws.Demos.Demo010_SelfPlay_AlphaZero.othello.demo_5.Small.app_info import app_info
from ws.RLAgents.E_SelfPlay.DeepLearning_SelfPlay.agent_mgt import agent_mgt

if __name__ == "__main__":
    agent_mgt( __file__). \
        fn_change_args({
            'NUM_MC_SIMULATIONS': 50,
            'NUM_TEST_GAMES': 12,
        }). \
        fn_show_args(). \
        fn_test_against_greedy()