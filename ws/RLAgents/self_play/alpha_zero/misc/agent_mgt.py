import inspect
import logging
import math
import os
import signal
# import timeit
from collections import namedtuple
from time import time
from datetime import datetime as dt

import numpy

from ws.RLAgents.self_play.alpha_zero.play.Arena import Arena
from ws.RLAgents.self_play.alpha_zero.play.GreedyPlayer import GreedyPlayer
from ws.RLAgents.self_play.alpha_zero.play.HumanPlayer import HumanPlayer
from ws.RLAgents.self_play.alpha_zero.play.RandomPlayer import RandomPlayer
from ws.RLAgents.self_play.alpha_zero.search.MctsSelector import MctsSelector

from ws.RLAgents.self_play.alpha_zero.train.coach import coach
from ws.RLEnvironments.self_play_games.othello.OthelloGame import OthelloGame as Game, OthelloGame
from ws.RLAgents.self_play.alpha_zero.train.NeuralNetWrapper import NeuralNetWrapper
from ws.RLUtils.common.AppInfo import AppInfo
from ws.RLUtils.monitoring.tracing.Recorder import Recorder
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgr
from ws.RLUtils.monitoring.tracing.tracer import tracer


def agent_mgr(args, file_path):

    log = logging.getLogger(__name__)
    args = args
    args.demo_folder, args.demo_name = AppInfo.fn_get_path_and_app_name(file_path)
    args.mcts_recursive = AppInfo.fn_arg_as_bool(args, 'mcts_recursive')
    game = OthelloGame(args.board_size)

    current_dir = file_path.rsplit('/', 1)[0]
    archive_dir = current_dir.replace('/Demos/', '/Archive/')
    args.archive_dir = archive_dir
    args.fn_record = log_mgr(log_dir=archive_dir, fixed_log_file=True)
    start_time = time()
    args.recorder = Recorder(args.fn_record)

    def exit_gracefully(signum, frame):
        #
        # if services.chart is not None:
        #     services.chart.fn_close()
        #     services.fn_record('@@@ Chart Saved')
        #
        # fn_archive_it()
        #
        # services.fn_record('TERMINATED EARLY AFTER SAVING MODEL WEIGHTS')
        # services.fn_record(f'Total Time Taken = {time() - start_time} seconds')
        exit()

    @tracer(args)
    def fn_train():
        signal.signal(signal.SIGINT, exit_gracefully)
        # args.fn_record('Loading %s...', Game.__name__)

        # args.fn_record('Loading %s...', NeuralNetWrapper.__name__)
        nnet = NeuralNetWrapper(args, game)

        if args.load_model:
            args.fn_record('Loading checkpoint "%s/%s"...', args.load_folder_file)
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        else:
            log.warning('Not loading a checkpoint!')

        # args.fn_record('  Loading the Coach...')
        c = coach(game, nnet, args)

        if args.load_model:
            # args.fn_record("  Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        # args.fn_record('  Starting the learning process ')
        c.fn_learn()

        return ret_refs

    @tracer(args)
    def fn_test_against_human():
        fn_human_player_policy = lambda g: HumanPlayer(g).play
        fn_test(fn_human_player_policy, verbose= True)
        return ret_refs

    @tracer(args)
    def fn_test_againt_random():
        fn_random_player_policy = lambda g: RandomPlayer(g).play
        fn_test(fn_random_player_policy, num_of_test_games= args.num_of_test_games)
        return ret_refs

    @tracer(args)
    def fn_test_against_greedy():
        fn_random_player_policy = lambda g: GreedyPlayer(g).play
        fn_test(fn_random_player_policy, num_of_test_games= args.num_of_test_games)
        return ret_refs

    def fn_test(fn_player_policy, verbose= False, num_of_test_games=2):
        signal.signal(signal.SIGINT, exit_gracefully)
        system_nn = NeuralNetWrapper(args, game)
        system_nn.load_checkpoint('tmp.1/', 'best.pth.tar')
        # args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        system_mcts = MctsSelector(game, system_nn, args)
        fn_system_policy = lambda x: numpy.argmax(system_mcts.getActionProb(x, temp=0))
        fn_contender_policy = fn_player_policy(game)
        arena = Arena(fn_system_policy, fn_contender_policy, game, display=OthelloGame.display)
        system_wins, system_losses, draws = arena.playGames(args.num_of_test_games, verbose=verbose)
        # args.fn_record(f'pwins:{pwins} nwins:{nwins} draws:{draws}')
        args.recorder.fn_record_message(f'wins:{system_wins} losses:{system_losses} draws:{draws}')

    @tracer(args)
    def fn_change_args(change_args):
        if change_args is not None:
            for k,v in change_args.items():
                change_args[k] = v
                # args.fn_record(f'  args[{k}] = {v}')
                args.recorder.fn_record_message(f'  x[{k}] = {v}')

        return ret_refs

    @tracer(args)
    def fn_show_args():

        for k,v in args.items():
            # args.fn_record(f'  args[{k}] = {v}')
            args.recorder.fn_record_message(f'  args[{k}] = {v}')

        return ret_refs

    @tracer(args)
    def fn_measure_time_elapsed():
        end_time = time()
        time_diff = int(end_time - start_time)
        mins = math.floor(time_diff / 60)
        secs = time_diff % 60
        args.recorder.fn_record_message(f'Time elapsed:    minutes: {mins}    seconds: {secs}')

        return ret_refs

    @tracer(args)
    def fn_archive_log_file():
        log_file_name = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
        dst_file_path = os.path.join(args.archive_dir, log_file_name)
        src_file_name = os.path.join(args.archive_dir, 'log.txt')
        os.rename(src_file_name, dst_file_path)
        return ret_refs

    ret_refs = namedtuple('_', ['fn_train','fn_test_against_human' ,'fn_test_againt_random' ,'fn_test_against_greedy' ,'fn_change_args' ,'fn_show_args' ,'fn_measure_time_elapsed' ,'fn_archive_log_file'])

    ret_refs.fn_train = fn_train
    ret_refs.fn_test_against_human = fn_test_against_human
    ret_refs.fn_test_againt_random = fn_test_againt_random
    ret_refs.fn_test_against_greedy = fn_test_against_greedy
    ret_refs.fn_change_args = fn_change_args
    ret_refs.fn_show_args = fn_show_args
    ret_refs.fn_measure_time_elapsed = fn_measure_time_elapsed
    ret_refs.fn_archive_log_file = fn_archive_log_file

    return ret_refs
