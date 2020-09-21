import inspect
import logging
import math
import os
import signal
# import timeit
from time import time
from datetime import datetime as dt

import numpy

from ws.RLAgents.self_play.alpha_zero.play.Arena import Arena
from ws.RLAgents.self_play.alpha_zero.play.GreedyPlayer import GreedyPlayer
from ws.RLAgents.self_play.alpha_zero.play.HumanPlayer import HumanPlayer
from ws.RLAgents.self_play.alpha_zero.play.RandomPlayer import RandomPlayer
from ws.RLAgents.self_play.alpha_zero.search.MctsSelector import MctsSelector

from ws.RLAgents.self_play.alpha_zero.train.Coach import Coach
from ws.RLEnvironments.self_play_games.othello.OthelloGame import OthelloGame as Game, OthelloGame
from ws.RLAgents.self_play.alpha_zero.train.NeuralNetWrapper import NeuralNetWrapper
from ws.RLUtils.common.AppInfo import AppInfo
from ws.RLUtils.monitoring.tracing.Recorder import Recorder
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgr

class Agent():

    def __init__(self, args, file_path):
        self.log = logging.getLogger(__name__)
        self.args = args
        self.args.demo_folder, self.args.demo_name = AppInfo.fn_get_path_and_app_name(file_path)
        self.args.mcts_recursive = AppInfo.fn_arg_as_bool(self.args, 'mcts_recursive')
        self.game = OthelloGame(self.args.board_size)

        current_dir = file_path.rsplit('/', 1)[0]
        archive_dir = current_dir.replace('/Demos/', '/Archive/')
        self.args.archive_dir = archive_dir
        self.args.fn_record = log_mgr(log_dir=archive_dir, fixed_log_file=True)
        self.start_time = time()
        self.args.recorder = Recorder(self.args.fn_record)

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


    def fn_train(self):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])

        signal.signal(signal.SIGINT, self.exit_gracefully)
        # self.args.fn_record('Loading %s...', Game.__name__)

        # self.args.fn_record('Loading %s...', NeuralNetWrapper.__name__)
        nnet = NeuralNetWrapper(self.args, self.game)

        if self.args.load_model:
            self.args.fn_record('Loading checkpoint "%s/%s"...', self.args.load_folder_file)
            nnet.load_checkpoint(self.args.load_folder_file[0], self.args.load_folder_file[1])
        else:
            self.log.warning('Not loading a checkpoint!')

        # self.args.fn_record('  Loading the Coach...')
        c = Coach(self.game, nnet, self.args)

        if self.args.load_model:
            # self.args.fn_record("  Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        # self.args.fn_record('  Starting the learning process ')
        c.fn_learn()

        self.args.recorder.fn_record_func_title_end()
        return self

    def fn_test_against_human(self):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])

        fn_human_player_policy = lambda g: HumanPlayer(g).play
        self.fn_test(fn_human_player_policy, verbose= True)

        self.args.recorder.fn_record_func_title_end()
        return self


    def fn_test_againt_random(self):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])

        fn_random_player_policy = lambda g: RandomPlayer(g).play
        self.fn_test(fn_random_player_policy, num_of_test_games= self.args.num_of_test_games)

        self.args.recorder.fn_record_func_title_end()
        return self

    def fn_test_against_greedy(self):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])

        fn_random_player_policy = lambda g: GreedyPlayer(g).play
        self.fn_test(fn_random_player_policy, num_of_test_games= self.args.num_of_test_games)

        self.args.recorder.fn_record_func_title_end()
        return self

    def fn_test(self, fn_player_policy, verbose= False, num_of_test_games=2):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        system_nn = NeuralNetWrapper(self.args, self.game)
        system_nn.load_checkpoint('tmp/', 'best.pth.tar')
        # args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        system_mcts = MctsSelector(self.game, system_nn, self.args)
        fn_system_policy = lambda x: numpy.argmax(system_mcts.getActionProb(x, temp=0))
        fn_contender_policy = fn_player_policy(self.game)
        arena = Arena(fn_system_policy, fn_contender_policy, self.game, display=OthelloGame.display)
        system_wins, system_losses, draws = arena.playGames(self.args.num_of_test_games, verbose=verbose)
        # self.args.fn_record(f'pwins:{pwins} nwins:{nwins} draws:{draws}')
        self.args.recorder.fn_record_message(f'wins:{system_wins} losses:{system_losses} draws:{draws}')

    def fn_change_args(self, args):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])

        if args is not None:
            for k,v in args.items():
                self.args[k] = v
                # self.args.fn_record(f'  args[{k}] = {v}')
                self.args.recorder.fn_record_message(f'  args[{k}] = {v}')

        self.args.recorder.fn_record_func_title_end()
        return self

    def fn_show_args(self):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])

        for k,v in self.args.items():
            # self.args.fn_record(f'  args[{k}] = {v}')
            self.args.recorder.fn_record_message(f'  args[{k}] = {v}')

        self.args.recorder.fn_record_func_title_end()
        return self

    def fn_measure_time_elapsed(self):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])

        end_time = time()
        time_diff = int(end_time - self.start_time)
        mins = math.floor(time_diff / 60)
        secs = time_diff % 60
        self.args.recorder.fn_record_message(f'Time elapsed:    minutes: {mins}    seconds: {secs}')

        self.args.recorder.fn_record_func_title_end()
        return self

    def fn_archive_log_file(self):
        log_file_name = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
        dst_file_path = os.path.join(self.args.archive_dir, log_file_name)
        src_file_name = os.path.join(self.args.archive_dir, 'log.txt')
        os.rename(src_file_name, dst_file_path)
        return self
