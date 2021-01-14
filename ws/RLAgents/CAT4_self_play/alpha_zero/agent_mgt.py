import math
import os
import shutil
import signal
import sys
from collections import namedtuple
from datetime import datetime as dt
from time import time

import numpy

from ws.RLAgents.CAT4_self_play.alpha_zero.misc.neural_net_mgt import neural_net_mgt
from ws.RLAgents.CAT4_self_play.alpha_zero.play.greedy_player_mgt import greedy_player_mgt
from ws.RLAgents.CAT4_self_play.alpha_zero.play.animated_player_mgt import animated_player_mgt
from ws.RLAgents.CAT4_self_play.alpha_zero.play.random_player_mgt import random_player_mgt
from ws.RLAgents.CAT4_self_play.alpha_zero.play.playground_mgt import playground_mgt
from ws.RLAgents.algo_lib.logic.search.monte_carlo_tree_search_mgt import monte_carlo_tree_search_mgt
from ws.RLAgents.CAT4_self_play.alpha_zero.train.training_mgt import training_mgt
from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt

from ws.RLUtils.monitoring.tracing.tracer import tracer
# from ws.RLUtils.setup.args_mgt import args_mgt
from ws.RLUtils.setup.startup_mgt import startup_mgt


def fn_setup_essential_managers(args_):
    args_.game_mgr = game_mgt(args_.BOARD_SIZE)
    args_.neural_net_mgr = neural_net_mgt(args_.game_mgr, args_.RESULTS_PATH_, args_.MODEL_NAME)

    args_.training_mgr = training_mgt(args_.neural_net_mgr, args_)
    return args_

def agent_mgt(file_path):
    # try:
        # args = args_mgt(file_path)
    args = startup_mgt(file_path)
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

    @tracer(args, verboscity= 4)
    def fn_train():
        nonlocal args

        signal.signal(signal.SIGINT, exit_gracefully)

        fn_setup_essential_managers(args)
        args.training_mgr.fn_execute_training_iterations()

        return agent_mgr

    @tracer(args)
    def fn_test_against_human():
        fn_human_player_policy = lambda g: animated_player_mgt(g)
        fn_test(args, fn_human_player_policy, verbose=True, NUM_TEST_GAMES=2)
        return agent_mgr

    @tracer(args, verboscity= 4)
    def fn_test_against_random():
        fn_random_player_policy = lambda g: random_player_mgt(g)
        fn_test(args, fn_random_player_policy, NUM_TEST_GAMES=args.NUM_TEST_GAMES)
        return agent_mgr

    @tracer(args, verboscity= 4)
    def fn_test_against_greedy():
        fn_random_player_policy = lambda g: greedy_player_mgt(g)
        fn_test(args, fn_random_player_policy, NUM_TEST_GAMES=args.NUM_TEST_GAMES)
        return agent_mgr

    def fn_test(args, fn_player_policy, verbose=False, NUM_TEST_GAMES=2):
        fn_setup_essential_managers(args)

        signal.signal(signal.SIGINT, exit_gracefully)
        system_nn = neural_net_mgt(args.game_mgr, args.RESULTS_PATH_, args.MODEL_NAME)
        if not system_nn.fn_load_model():
            return

        system_mcts = monte_carlo_tree_search_mgt(args.game_mgr, system_nn, args)
        fn_system_policy = lambda state: numpy.argmax(system_mcts.fn_get_policy(state, do_random_selection=False))
        fn_contender_policy = fn_player_policy(args.game_mgr)
        playground = playground_mgt(fn_system_policy, fn_contender_policy, args.game_mgr,
                                    fn_display=game_mgt(args['BOARD_SIZE']).fn_display,
                                    msg_recorder=args.CALL_TRACER_.fn_write)
        system_wins, system_losses, draws = playground.fn_play_games(NUM_TEST_GAMES, verbose=verbose)

        args.CALL_TRACER_.fn_write(f'wins:{system_wins} losses:{system_losses} draws:{draws}')

    @tracer(args, verboscity= 4)
    def fn_reset():
        if os.path.exists(args.RESULTS_REL_PATH):
            shutil.rmtree(args.RESULTS_REL_PATH)

        return agent_mgr

    @tracer(args, verboscity= 4)
    def fn_change_args(change_args):
        if change_args is not None:
            for k, v in change_args.items():
                args[k] = v
                args.CALL_TRACER_.fn_write(f'  args_[{k}] = {v}')
        agent_mgr.args = args
        return agent_mgr

    @tracer(args, verboscity= 4)
    def fn_show_args():
        for k, v in args.items():
            args.CALL_TRACER_.fn_write(f'  args_[{k}] = {v}')
        return agent_mgr

    @tracer(args, verboscity= 4)
    def fn_measure_time_elapsed():
        nonlocal start_time
        end_time = time()
        time_diff = int(end_time - start_time)
        mins = math.floor(time_diff / 60)
        secs = time_diff % 60
        args.CALL_TRACER_.fn_write(f'Time elapsed:    minutes: {mins}    seconds: {secs}')
        start_time = time()
        return agent_mgr

    @tracer(args, verboscity= 4)
    def fn_archive_log_file():
        dst_subfolder_rel_path = dt.now().strftime("%Y_%m_%d_%H_%M_%S")

        dst_full_path = os.path.join(args.ARCHIVE_DIR_, dst_subfolder_rel_path)
        os.mkdir(dst_full_path)

        if os.path.exists(args.OLD_MODEL_FILEPATH_):
            shutil.copy(args.OLD_MODEL_FILEPATH_, dst_full_path)

        # copy model.tar
        if os.path.exists(args.MODEL_FILEPATH_):
            shutil.copy(args.MODEL_FILEPATH_, dst_full_path)

        return agent_mgr

    start_time = time()
    if os.path.exists(args.MODEL_FILEPATH_):
        shutil.copy(args.MODEL_FILEPATH_, args.OLD_MODEL_FILEPATH_)

    agent_mgr = namedtuple('_',
                           ['fn_reset', 'fn_train', 'fn_test_against_human', 'fn_test_againt_random', 'fn_test_against_greedy',
                            'fn_change_args', 'fn_show_args', 'fn_measure_time_elapsed', 'fn_archive_log_file',
                            'args_'])
    agent_mgr.fn_reset = fn_reset
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

    # except Exception as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
    #     raise e
