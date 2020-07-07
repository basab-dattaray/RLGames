from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.ARGS import args
from ws.RLAgents.self_play.alpha_zero.play import Arena
from ws.RLAgents.self_play.alpha_zero.play.HumanPlayer import HumanPlayer
from ws.RLAgents.self_play.alpha_zero.search.recursive.MCTS import MCTS
from othello.OthelloGame import OthelloGame
from ws.RLAgents.self_play.alpha_zero.play.RandomPlayer import RandomPlayer
from ws.RLAgents.self_play.alpha_zero.play.GreedyPlayer import GreedyPlayer
from ws.RLAgents.self_play.alpha_zero._game.othello._ml_lib.pytorch.NNet import NNetWrapper

from ws.RLUtils.common.AppInfo import AppInfo

import numpy as np
from ws.RLAgents.self_play.alpha_zero.misc.utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

dir_path, app_name = AppInfo.fn_get_path_and_app_name(__file__)

g = OthelloGame(5)


# all players
rp = RandomPlayer(g).play
gp = GreedyPlayer(g).play
hp = HumanPlayer(g).play



# nnet players
n1 = NNetWrapper(args, g)



n1.load_checkpoint('tmp/','best.pth.tar')

args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNetWrapper(g)
    n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=OthelloGame.display)

print(arena.playGames(2, verbose=True))
