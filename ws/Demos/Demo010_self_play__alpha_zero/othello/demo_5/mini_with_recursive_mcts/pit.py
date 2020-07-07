from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.ARGS import args
from ws.RLAgents.self_play.alpha_zero.play import Arena
from ws.RLAgents.self_play.alpha_zero.play.HumanPlayer import HumanPlayer
from ws.RLAgents.self_play.alpha_zero.search.recursive.MCTS import MCTS
from ws.RLEnvironments.self_play_games.othello.OthelloGame import OthelloGame
from ws.RLAgents.self_play.alpha_zero.play.RandomPlayer import RandomPlayer
from ws.RLAgents.self_play.alpha_zero.play.GreedyPlayer import GreedyPlayer
from ws.RLAgents.self_play.alpha_zero.train.NeuralNetWrapper import NeuralNetWrapper

from ws.RLUtils.common.AppInfo import AppInfo

import numpy as np

fn_random_player_policy = lambda g: RandomPlayer(g).play
fn_greedy_player_policy = lambda g: GreedyPlayer(g).play
fn_human_player_policy = lambda g: HumanPlayer(g).play

dir_path, app_name = AppInfo.fn_get_path_and_app_name(__file__)

game = OthelloGame(args.board_size)
system_nn = NeuralNetWrapper(args, game)

system_nn.load_checkpoint('tmp/', 'best.pth.tar')

# args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
system_mcts = MCTS(game, system_nn, args)
fn_system_policy = lambda x: np.argmax(system_mcts.getActionProb(x, temp=0))

fn_contender_policy = fn_human_player_policy(game)

arena = Arena.Arena(fn_system_policy, fn_contender_policy, game, display=OthelloGame.display)

print(arena.playGames(2, verbose=True))
