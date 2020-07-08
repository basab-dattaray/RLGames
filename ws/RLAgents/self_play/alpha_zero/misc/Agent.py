import logging
import signal

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
from ws.RLUtils.decorators.breadcrumbs import encapsulate

class Agent():
    @classmethod
    def fn_init(cls, args, file_path):
        return Agent(args, file_path)

    def __init__(self, args, file_path):
        self.log = logging.getLogger(__name__)
        self.args = args
        self.args.demo_folder, self.args.demo_name = AppInfo.fn_get_path_and_app_name(file_path)
        self.args.mcts_recursive = AppInfo.fn_arg_as_bool(self.args, 'mcts_recursive')
        self.game = OthelloGame(self.args.board_size)

    def exit_gracefully(self, signum, frame):
        #
        # if self.services.chart is not None:
        #     self.services.chart.fn_close()
        #     self.services.fn_record('@@@ Chart Saved')
        #
        # self.fn_archive_it()
        #
        # self.services.fn_record('TERMINATED EARLY AFTER SAVING MODEL WEIGHTS')
        # self.services.fn_record(f'Total Time Taken = {time() - self.start_time} seconds')
        exit()

    @encapsulate
    def fn_train(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        self.log.info('Loading %s...', Game.__name__)

        self.log.info('Loading %s...', NeuralNetWrapper.__name__)
        nnet = NeuralNetWrapper(self.args, self.game)

        if self.args.load_model:
            self.log.info('Loading checkpoint "%s/%s"...', self.args.load_folder_file)
            nnet.load_checkpoint(self.args.load_folder_file[0], self.args.load_folder_file[1])
        else:
            self.log.warning('Not loading a checkpoint!')

        self.log.info('Loading the Coach...')
        c = Coach(self.game, nnet, self.args)

        if self.args.load_model:
            self.log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        self.log.info('Starting the learning process 🎉')
        c.learn()
        return self

    @encapsulate
    def fn_test_against_human(self):
        fn_human_player_policy = lambda g: HumanPlayer(g).play
        self.fn_test(fn_human_player_policy, verbose= True)
        return self

    @encapsulate
    def fn_test_againt_random(self):
        fn_random_player_policy = lambda g: RandomPlayer(g).play
        self.fn_test(fn_random_player_policy, num_of_test_games= self.args.num_of_test_games)
        return self

    @encapsulate
    def fn_test_against_greedy(self):
        fn_random_player_policy = lambda g: GreedyPlayer(g).play
        self.fn_test(fn_random_player_policy, num_of_test_games= self.args.num_of_test_games)
        return self

    def fn_test(self, fn_player_policy, verbose= False, num_of_test_games=2):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        system_nn = NeuralNetWrapper(self.args, self.game)
        system_nn.load_checkpoint('tmp/', 'best.pth.tar')
        # args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        system_mcts = MCTS(self.game, system_nn, self.args)
        fn_system_policy = lambda x: numpy.argmax(system_mcts.getActionProb(x, temp=0))
        fn_contender_policy = fn_player_policy(self.game)
        arena = Arena(fn_system_policy, fn_contender_policy, self.game, display=OthelloGame.display)
        print(arena.playGames(num_of_test_games, verbose=verbose))

    @encapsulate
    def fn_change_args(self, args):
        if args is not None:
            for k,v in args.items():
                self.args[k] = v
                print(f'args[{k}] = {v}')
        return self

    @encapsulate
    def fn_show_args(self):
        for k,v in self.args.items():
            print(f'args[{k}] = {v}')

        return self