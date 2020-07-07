import logging

import numpy

from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.ARGS import args
from ws.RLAgents.self_play.alpha_zero.play.Arena import Arena
from ws.RLAgents.self_play.alpha_zero.play.GreedyPlayer import GreedyPlayer
from ws.RLAgents.self_play.alpha_zero.play.HumanPlayer import HumanPlayer
from ws.RLAgents.self_play.alpha_zero.play.RandomPlayer import RandomPlayer
from ws.RLAgents.self_play.alpha_zero.search.recursive.MCTS import MCTS
from ws.RLAgents.self_play.alpha_zero.train.Coach import Coach
from ws.RLEnvironments.self_play_games.othello.OthelloGame import OthelloGame as Game, OthelloGame
from ws.RLAgents.self_play.alpha_zero.train.NeuralNetWrapper import NeuralNetWrapper
from ws.RLUtils.common.AppInfo import AppInfo


class Agent():
    def __init__(self, args, file_path):
        self.log = logging.getLogger(__name__)
        self.args = args
        self.args.demo_folder, self.args.demo_name = AppInfo.fn_get_path_and_app_name(file_path)
        self.game = OthelloGame(args.board_size)

    def fn_train(self):
        self.log.info('Loading %s...', Game.__name__)

        self.log.info('Loading %s...', NeuralNetWrapper.__name__)
        nnet = NeuralNetWrapper(args, self.game)

        if args.load_model:
            self.log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        else:
            self.log.warning('Not loading a checkpoint!')

        self.log.info('Loading the Coach...')
        c = Coach(self.game, nnet, args)

        if args.load_model:
            self.log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        self.log.info('Starting the learning process 🎉')
        c.learn()

    def fn_test_human(self):
        fn_random_player_policy = lambda g: RandomPlayer(g).play
        fn_greedy_player_policy = lambda g: GreedyPlayer(g).play
        fn_human_player_policy = lambda g: HumanPlayer(g).play
        dir_path, app_name = AppInfo.fn_get_path_and_app_name(__file__)

        system_nn = NeuralNetWrapper(args, self.game)
        system_nn.load_checkpoint('tmp/', 'best.pth.tar')
        # args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        system_mcts = MCTS(self.game, system_nn, args)
        fn_system_policy = lambda x: numpy.argmax(system_mcts.getActionProb(x, temp=0))
        fn_contender_policy = fn_human_player_policy(self.game)
        arena = Arena(fn_system_policy, fn_contender_policy, self.game, display=OthelloGame.display)
        print(arena.playGames(2, verbose=True))