
import coloredlogs

from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.ARGS import args
from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.Agent import Agent

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


if __name__ == "__main__":
    agent = Agent(args, __file__)
    agent.fn_train()