import logging
import math
import os
import signal
import sys
from collections import namedtuple

from shutil import copy, move
from time import time
from datetime import datetime as dt

import numpy

from ws.RLUtils.common.DotDict import DotDict
from ws.RLAgents.self_play.alpha_zero.play.playground_mgt import playground_mgt
from ws.RLAgents.self_play.alpha_zero.play.GreedyPlayer import GreedyPlayer
from ws.RLAgents.self_play.alpha_zero.play.HumanPlayer import HumanPlayer
from ws.RLAgents.self_play.alpha_zero.play.RandomPlayer import RandomPlayer
from ws.RLAgents.self_play.alpha_zero.search.mcts_adapter import mcts_adapter

from ws.RLAgents.self_play.alpha_zero.train.training_mgt import training_mgt
from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt
from ws.RLAgents.self_play.alpha_zero.misc.neural_net_mgt import neural_net_mgt
from ws.RLUtils.common.AppInfo import AppInfo
from ws.RLUtils.monitoring.tracing.call_trace_mgt import call_trace_mgt
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt
from ws.RLUtils.monitoring.tracing.tracer import tracer


def fn_init_arg_with_default_val(arguments, name, val):
    arguments = DotDict(arguments.copy())
    arguments[name] = val
    return arguments

def agent_mgt(args, file_path):

   try:
        def _fn_setup(file_path):
            def _fn_setup_training_mgr(args_):
                args_ = fn_init_arg_with_default_val(args_, 'num_of_successes_for_model_upgrade', 1)
                args_ = fn_init_arg_with_default_val(args_, 'run_recursive_search', True)

                args_ = fn_init_arg_with_default_val(args_, 'mcts_ucb_use_log_in_numerator', True)
                args_ = fn_init_arg_with_default_val(args_, 'mcts_ucb_use_action_prob_for_exploration', True)
                args_ = fn_init_arg_with_default_val(args_, 'do_load_model', True)
                args_ = fn_init_arg_with_default_val(args_, 'do_load_samples', False)

                args_ = fn_init_arg_with_default_val(args_, 'training_mgr', training_mgt(args_.neural_net_mgr, args_))
                return args_

            def _fn_set_default_args(args, file_path):
                args_copy = fn_init_arg_with_default_val(args, 'logger', logging.getLogger(__name__))
                demo_folder, demo_name = AppInfo.fn_get_path_and_app_name(file_path)
                args_copy = _fn_general_args_init(args_copy, demo_folder, demo_name, file_path)
                args_copy = _fn_setup_training_mgr(args_copy)
                # training_mgr = training_mgt(args_copy.neural_net_mgr, args_copy)
                return args_copy

            def _fn_general_args_init(args_, demo_folder, demo_name, file_path):
                args_ = fn_init_arg_with_default_val(args_, 'demo_folder', demo_folder)
                args_ = fn_init_arg_with_default_val(args_, 'demo_name', demo_name)
                args_ = fn_init_arg_with_default_val(args_, 'model_name', 'model.tar')
                args_ = fn_init_arg_with_default_val(args_, 'rel_model_path', 'model/')
                args_ = fn_init_arg_with_default_val(args_, 'temp_model_exchange_filename', '_tmp.tar')
                current_dir = file_path.rsplit('/', 1)[0]
                archive_dir = current_dir.replace('/Demos/', '/Archives/')
                args_ = fn_init_arg_with_default_val(args_, 'archive_dir', archive_dir)
                args_ = fn_init_arg_with_default_val(args_, 'fn_record',
                                                     log_mgt(log_dir=archive_dir, fixed_log_file=True))
                args_ = fn_init_arg_with_default_val(args_, 'calltracer', call_trace_mgt(args_.fn_record))
                src_model_folder = os.path.join(args_.demo_folder, args_.rel_model_path)
                args_ = fn_init_arg_with_default_val(args_, 'src_model_file_path',
                                                     os.path.join(src_model_folder, args_.model_name))
                args_ = fn_init_arg_with_default_val(args_, 'old_model_file_path',
                                                     os.path.join(src_model_folder, 'old_' + args_.model_name))
                args_ = fn_init_arg_with_default_val(args_, 'rel_model_path', 'model/')

                args_ = fn_init_arg_with_default_val(args_, 'game_mgr', game_mgt(args_.board_size))
                args_.neural_net_mgr = neural_net_mgt(args_)
                args_ = fn_init_arg_with_default_val(args_, 'neural_net_mgr', args_.neural_net_mgr)
                return args_

            arguments = _fn_set_default_args(args, file_path)

            return arguments

        args = _fn_setup(file_path)

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

            args.training_mgr.fn_execute_training_iterations()

            return agent_mgr

        @tracer(args)
        def fn_test_against_human():
            fn_human_player_policy = lambda g: HumanPlayer(g).fn_get_action
            fn_test(fn_human_player_policy, verbose= True, num_of_test_games= 2)
            return agent_mgr

        @tracer(args)
        def fn_test_against_random():
            fn_random_player_policy = lambda g: RandomPlayer(g).fn_get_action
            fn_test(fn_random_player_policy, num_of_test_games= args.num_of_test_games)
            return agent_mgr

        @tracer(args)
        def fn_test_against_greedy():
            fn_random_player_policy = lambda g: GreedyPlayer(g).fn_get_action
            fn_test(fn_random_player_policy, num_of_test_games= args.num_of_test_games)
            return agent_mgr

        def fn_test(fn_player_policy, verbose= False, num_of_test_games= 2):
            signal.signal(signal.SIGINT, exit_gracefully)
            system_nn = neural_net_mgt(args)
            if not system_nn.fn_load_model():
                return

            system_mcts = mcts_adapter(system_nn, args)
            fn_system_policy = lambda state: numpy.argmax(system_mcts.fn_get_policy(state, do_random_selection=False))
            fn_contender_policy = fn_player_policy(args.game_mgr)
            playground = playground_mgt(fn_system_policy, fn_contender_policy, args.game_mgr, fn_display=game_mgt(args['board_size']).fn_display,
                          msg_recorder=args.calltracer.fn_write)
            system_wins, system_losses, draws = playground.fn_play_games(num_of_test_games, verbose=verbose)

            args.calltracer.fn_write(f'wins:{system_wins} losses:{system_losses} draws:{draws}')

        @tracer(args)
        def fn_change_args(change_args):
            if change_args is not None:
                for k,v in change_args.items():
                    args[k] = v
                    # nn_args.fn_record(f'  nn_args[{k}] = {v}')
                    args.calltracer.fn_write(f'  args_[{k}] = {v}')
            agent_mgr.args = args
            return agent_mgr

        @tracer(args)
        def fn_show_args():

            for k,v in args.items():
                # nn_args.fn_record(f'  nn_args[{k}] = {v}')
                args.calltracer.fn_write(f'  args_[{k}] = {v}')

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
                                     'args_'])

        agent_mgr.fn_train = fn_train
        agent_mgr.fn_test_against_human = fn_test_against_human
        agent_mgr.fn_test_against_random = fn_test_against_random
        agent_mgr.fn_test_against_greedy = fn_test_against_greedy
        agent_mgr.fn_change_args = fn_change_args
        agent_mgr.fn_show_args = fn_show_args
        agent_mgr.fn_measure_time_elapsed = fn_measure_time_elapsed
        agent_mgr.fn_archive_log_file = fn_archive_log_file
        agent_mgr.arguments = args
        return agent_mgr
   except Exception as e:
       exc_type, exc_obj, exc_tb = sys.exc_info()
       fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
       print(exc_type, fname, exc_tb.tb_lineno)
       raise e


