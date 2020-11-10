import inspect
import logging
import math
import os
import signal
# import timeit
from collections import namedtuple
from copy import deepcopy
from shutil import copy, move
from time import time
from datetime import datetime as dt

import numpy

from ws.RLAgents.self_play.alpha_zero.misc.utils import dotdict
from ws.RLAgents.self_play.alpha_zero.play.playground_mgt import playground_mgt
from ws.RLAgents.self_play.alpha_zero.play.GreedyPlayer import GreedyPlayer
from ws.RLAgents.self_play.alpha_zero.play.HumanPlayer import HumanPlayer
from ws.RLAgents.self_play.alpha_zero.play.RandomPlayer import RandomPlayer
from ws.RLAgents.self_play.alpha_zero.search.mcts_adapter import mcts_adapter

from ws.RLAgents.self_play.alpha_zero.train.training_mgt import training_mgt
from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt as Game, game_mgt
from ws.RLAgents.self_play.alpha_zero.train.neural_net_mgt import neural_net_mgt
from ws.RLUtils.common.AppInfo import AppInfo
from ws.RLUtils.monitoring.tracing.call_trace_mgt import call_trace_mgt
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt
from ws.RLUtils.monitoring.tracing.tracer import tracer


def agent_mgt(args, file_path):
    def setup(file_path):

        args_out = dotdict(args.copy())
        _fn_init_arg_with_default_val(args_out, 'logger',logging.getLogger(__name__))
        args_out.demo_folder, args_out.demo_name = AppInfo.fn_get_path_and_app_name(file_path)
        args_out.run_recursive_search = AppInfo.fn_arg_as_bool(args_out, 'run_recursive_search')
        args_out.game = game_mgt(args_out.board_size)
        _fn_init_arg_with_default_val(args_out, 'num_of_successes_for_model_upgrade', 1)
        _fn_init_arg_with_default_val(args_out, 'rel_model_path', 'model/')
        _fn_init_arg_with_default_val(args_out, 'do_load_model', False)
        _fn_init_arg_with_default_val(args_out, 'do_load_samples', False)
        _fn_init_arg_with_default_val(args_out, 'model_name', 'model.tar')
        _fn_init_arg_with_default_val(args_out, 'temp_model_exchange_name', '_tmp.tar')
        current_dir = file_path.rsplit('/', 1)[0]
        archive_dir = current_dir.replace('/Demos/', '/Archives/')
        args_out.archive_dir = archive_dir
        args_out.fn_record = log_mgt(log_dir=archive_dir, fixed_log_file=True)
        args_out.calltracer = call_trace_mgt(args_out.fn_record)
        src_model_folder = os.path.join(args_out.demo_folder, args_out.rel_model_path)
        args_out.src_model_file_path = os.path.join(src_model_folder, args_out.model_name)
        args_out.old_model_file_path = os.path.join(src_model_folder, 'old_' + args_out.model_name)


        return args_out

    args = setup(file_path)

    def exit_gracefully(signum, frame):
        #
        # if services.chart is not None:
        #     services.chart.fn_close()
        #     services.fn_log('@@@ Chart Saved')
        #
        # fn_archive_it()
        #
        # services.fn_log('TERMINATED EARLY AFTER SAVING MODEL WEIGHTS')
        # services.fn_log(f'Total Time Taken = {time() - start_time} seconds')
        exit()

    @tracer(args)

    def fn_train():
        signal.signal(signal.SIGINT, exit_gracefully)

        training_mgr = init_training_mgr()
        training_mgr.fn_execute_training_iterations()

        return agent_mgr

    def init_training_mgr():
        neural_net = neural_net_mgt(args)
        if args.do_load_model:
            # nn_args.fn_record('Loading rel_model_path "%state/%state"...', nn_args.load_folder_file)
            if not neural_net.fn_load_model():
                args.fn_record('*** unable to load model')
            else:
                args.fn_record('!!! loaded model')
        else:
            args.logger.warning('!!! Not loading a rel_model_path!')
        training_mgr = training_mgt(neural_net, args)
        return training_mgr

    @tracer(args)
    def fn_test_against_human():
        fn_human_player_policy = lambda g: HumanPlayer(g).play
        fn_test(fn_human_player_policy, verbose= True)
        return agent_mgr

    @tracer(args)
    def fn_test_againt_random():
        fn_random_player_policy = lambda g: RandomPlayer(g).play
        fn_test(fn_random_player_policy, num_of_test_games= args.num_of_test_games)
        return agent_mgr

    @tracer(args)
    def fn_test_against_greedy():
        fn_random_player_policy = lambda g: GreedyPlayer(g).play
        fn_test(fn_random_player_policy, num_of_test_games= args.num_of_test_games)
        return agent_mgr

    def fn_test(fn_player_policy, verbose= False, num_of_test_games=2):
        signal.signal(signal.SIGINT, exit_gracefully)
        system_nn = neural_net_mgt(args)
        if not system_nn.fn_load_model():
            return

        system_mcts = mcts_adapter(system_nn, args)
        fn_system_policy = lambda x: numpy.argmax(system_mcts.fn_get_action_probabilities(x, spread_probabilities=0))
        fn_contender_policy = fn_player_policy(args.game)
        arena = playground_mgt(fn_system_policy, fn_contender_policy, args.game, fn_display=game_mgt(args['board_size']).fn_display,
                      msg_recorder=args.calltracer.fn_write)
        system_wins, system_losses, draws = arena.fn_play_games(args.num_of_test_games, verbose=verbose)

        args.calltracer.fn_write(f'wins:{system_wins} losses:{system_losses} draws:{draws}')

    @tracer(args)
    def fn_change_args(change_args):
        if change_args is not None:
            for k,v in change_args.items():
                change_args[k] = v
                # nn_args.fn_record(f'  nn_args[{k}] = {v}')
                args.calltracer.fn_write(f'  x[{k}] = {v}')

        return agent_mgr

    @tracer(args)
    def fn_show_args():

        for k,v in args.items():
            # nn_args.fn_record(f'  nn_args[{k}] = {v}')
            args.calltracer.fn_write(f'  nn_args[{k}] = {v}')

        return agent_mgr

    @tracer(args)
    def fn_measure_time_elapsed():
        end_time = time()
        time_diff = int(end_time - start_time)
        mins = math.floor(time_diff / 60)
        secs = time_diff % 60
        args.calltracer.fn_write(f'Time elapsed:    minutes: {mins}    seconds: {secs}')

        return agent_mgr


    @tracer(args)
    def fn_archive_log_file():
        dst_subfolder_rel_path = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
        dst_full_path = os.path.join(args.archive_dir, dst_subfolder_rel_path)
        os.mkdir(dst_full_path)

        # move log.txt
        src_log_file_name = os.path.join(args.archive_dir, 'log.txt')
        if os.path.exists(src_log_file_name):
            move(src_log_file_name, dst_full_path)

        # move old_model.tar
        if os.path.exists(args.old_model_file_path):
            copy(args.old_model_file_path, dst_full_path)

        # copy model.tar
        if os.path.exists(args.src_model_file_path):
            copy(args.src_model_file_path, dst_full_path)

        return agent_mgr

    start_time = time()
    if os.path.exists(args.src_model_file_path):
        copy(args.src_model_file_path, args.old_model_file_path)

    agent_mgr = namedtuple('_', ['fn_train','fn_test_against_human' ,'fn_test_againt_random' ,'fn_test_against_greedy' ,'fn_change_args' ,'fn_show_args' ,'fn_measure_time_elapsed' ,'fn_archive_log_file',
                                 'nn_args'])

    agent_mgr.fn_train = fn_train
    agent_mgr.fn_test_against_human = fn_test_against_human
    agent_mgr.fn_test_againt_random = fn_test_againt_random
    agent_mgr.fn_test_against_greedy = fn_test_against_greedy
    agent_mgr.fn_change_args = fn_change_args
    agent_mgr.fn_show_args = fn_show_args
    agent_mgr.fn_measure_time_elapsed = fn_measure_time_elapsed
    agent_mgr.fn_archive_log_file = fn_archive_log_file
    agent_mgr.arguments = args
    return agent_mgr

def _fn_init_arg_with_default_val(args, name, val):
    if name not in args.keys():
        args[name] = val
