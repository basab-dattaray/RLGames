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
from ws.RLUtils.modelling.archive_mgt import archive_mgt

from ws.RLUtils.monitoring.tracing.tracer import tracer
# from ws.RLUtils.setup.args_mgt import args_mgt
from ws.RLUtils.setup.startup_mgt import startup_mgt


def fn_setup_essential_managers(app_info):
    app_info.game_mgr = game_mgt(app_info.BOARD_SIZE)
    app_info.neural_net_mgr = neural_net_mgt(app_info.game_mgr, app_info.RESULTS_PATH_)

    app_info.training_mgr = training_mgt(app_info.neural_net_mgr, app_info)
    return app_info

def agent_mgt(file_path):
    app_info = startup_mgt(file_path)
    app_info = fn_setup_essential_managers(app_info)

    fn_archive = archive_mgt(
        app_info.neural_net_mgr.fn_save_model,
        results_path= app_info.RESULTS_PATH_,
        archive_path= app_info.ARCHIVE_PATH_BEFORE_,
    )
    fn_archive()

    def exit_gracefully(signum, frame):
        app_info.fn_log('!!! TERMINATING EARLY!!!')
        archive_msg = fn_archive(archive_folder_path= app_info.ARCHIVE_PATH_AFTER_)
        app_info.fn_log(archive_msg)

        # chart.fn_close()
        app_info.ENV.fn_close()
        exit()

    @tracer(app_info, verboscity= 4)
    def fn_train():
        nonlocal app_info

        signal.signal(signal.SIGINT, exit_gracefully)

        app_info.training_mgr.fn_execute_training_iterations()

        return agent_mgr

    @tracer(app_info)
    def fn_test_against_human():
        fn_human_player_policy = lambda g: animated_player_mgt(g)
        fn_test(app_info, fn_human_player_policy, verbose=True, NUM_TEST_GAMES=2)
        return agent_mgr

    @tracer(app_info, verboscity= 4)
    def fn_test_against_random():
        fn_random_player_policy = lambda g: random_player_mgt(g)
        fn_test(app_info, fn_random_player_policy, NUM_TEST_GAMES=app_info.NUM_TEST_GAMES)
        return agent_mgr

    @tracer(app_info, verboscity= 4)
    def fn_test_against_greedy():
        fn_random_player_policy = lambda g: greedy_player_mgt(g)
        fn_test(app_info, fn_random_player_policy, NUM_TEST_GAMES=app_info.NUM_TEST_GAMES)
        return agent_mgr

    def fn_test(app_info, fn_player_policy, verbose=False, NUM_TEST_GAMES=2):
        signal.signal(signal.SIGINT, exit_gracefully)
        system_nn = neural_net_mgt(app_info.game_mgr, app_info.RESULTS_PATH_)
        if not system_nn.fn_load_model():
            return

        system_mcts = monte_carlo_tree_search_mgt(app_info.game_mgr, system_nn, app_info)
        fn_system_policy = lambda state: numpy.argmax(system_mcts.fn_get_policy(state, do_random_selection=False))
        fn_contender_policy = fn_player_policy(app_info.game_mgr)
        playground = playground_mgt(fn_system_policy, fn_contender_policy, app_info.game_mgr,
                                    fn_display=game_mgt(app_info['BOARD_SIZE']).fn_display,
                                    msg_recorder=app_info.CALL_TRACER_.fn_write)
        system_wins, system_losses, draws = playground.fn_play_games(NUM_TEST_GAMES, verbose=verbose)

        app_info.CALL_TRACER_.fn_write(f'wins:{system_wins} losses:{system_losses} draws:{draws}')

    @tracer(app_info, verboscity= 4)
    def fn_reset():
        if os.path.exists(app_info.RESULTS_REL_PATH):
            shutil.rmtree(app_info.RESULTS_REL_PATH)

        return agent_mgr

    @tracer(app_info, verboscity= 4)
    def fn_change_args(change_args):
        if change_args is not None:
            for k, v in change_args.items():
                app_info[k] = v
                app_info.CALL_TRACER_.fn_write(f'  app_info[{k}] = {v}')
        agent_mgr.app_info = app_info
        return agent_mgr

    @tracer(app_info, verboscity= 4)
    def fn_show_args():
        for k, v in app_info.items():
            app_info.CALL_TRACER_.fn_write(f'  app_info[{k}] = {v}')
        return agent_mgr

    @tracer(app_info, verboscity= 4)
    def fn_measure_time_elapsed():
        nonlocal start_time
        end_time = time()
        time_diff = int(end_time - start_time)
        mins = math.floor(time_diff / 60)
        secs = time_diff % 60
        app_info.CALL_TRACER_.fn_write(f'Time elapsed:    minutes: {mins}    seconds: {secs}')
        start_time = time()
        return agent_mgr

    @tracer(app_info, verboscity= 4)
    def fn_archive_log_file():
        archive_msg = fn_archive(archive_folder_path=app_info.ARCHIVE_PATH_AFTER_)
        app_info.fn_log(archive_msg)

        return agent_mgr

    start_time = time()
    # if os.path.exists(app_info.MODEL_FILEPATH_):
    #     shutil.copy(app_info.MODEL_FILEPATH_, app_info.OLD_MODEL_FILEPATH_)

    agent_mgr = namedtuple('_',
                           ['fn_reset', 'fn_train', 'fn_test_against_human', 'fn_test_againt_random', 'fn_test_against_greedy',
                            'fn_change_args', 'fn_show_args', 'fn_measure_time_elapsed', 'fn_archive_log_file',
                            'app_info'])
    agent_mgr.fn_reset = fn_reset
    agent_mgr.fn_train = fn_train
    agent_mgr.fn_test_against_human = fn_test_against_human
    agent_mgr.fn_test_against_random = fn_test_against_random
    agent_mgr.fn_test_against_greedy = fn_test_against_greedy
    agent_mgr.fn_change_args = fn_change_args
    agent_mgr.fn_show_args = fn_show_args
    agent_mgr.fn_measure_time_elapsed = fn_measure_time_elapsed
    agent_mgr.fn_archive_log_file = fn_archive_log_file
    agent_mgr.arguments = app_info
    return agent_mgr

    # except Exception as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
    #     raise e
