
import coloredlogs

from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.ARGS import args
from ws.RLAgents.self_play.alpha_zero.misc.agent_mgt import agent_mgr

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

#
# if __name__ == "__main__":
#     agent = Agent(args, __file__)
#     agent.fn_train()

if __name__ == "__main__":
    try:

        agent_mgr(args, __file__). \
        fn_show_args(). \
        fn_show_args(). \
        fn_show_args()

    except Exception as x:
        print(f'*** DEMO ---{x}')