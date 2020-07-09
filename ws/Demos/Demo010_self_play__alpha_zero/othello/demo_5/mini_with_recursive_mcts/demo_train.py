
import coloredlogs

from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.ARGS import args
from ws.RLAgents.self_play.alpha_zero.misc.Agent import Agent

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

#
# if __name__ == "__main__":
#     agent = Agent(args, __file__)
#     agent.fn_train()

if __name__ == "__main__":
    try:
        Agent.fn_init(args, __file__). \
            fn_change_args({
                'mcts_recursive': 1,
            }). \
            fn_train().\
            fn_change_args({
                'numMCTSSims': 50,
            }). \
            fn_show_args(). \
            fn_test_against_greedy(). \
            fn_test_againt_random()
    except Exception as x:
        print(f'*** DEMO ---{x}')