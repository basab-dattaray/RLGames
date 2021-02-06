import os
import shutil
from collections import namedtuple
from time import time

import numpy

# from ws.RLAgents.E_SelfPlay.model_mgt import model_mgt
from ws.RLAgents.E_SelfPlay.model_mgt import model_mgt
from ws.RLAgents.E_SelfPlay.play.greedy_player_mgt import greedy_player_mgt
from ws.RLAgents.E_SelfPlay.play.animated_player_mgt import animated_player_mgt
from ws.RLAgents.E_SelfPlay.play.random_player_mgt import random_player_mgt
from ws.RLAgents.E_SelfPlay.play.playground_mgt import playground_mgt
from ws.RLUtils.algo_lib.search.monte_carlo_tree_search_mgt import monte_carlo_tree_search_mgt
from ws.RLAgents.E_SelfPlay.train.training_mgt import training_mgt
from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt


from ws.RLUtils.monitoring.tracing.tracer import tracer

def agent_mgt(app_info, common_functions):
    game_mgr = game_mgt(app_info.BOARD_SIZE)
    neural_net_mgr = model_mgt(game_mgr, app_info.RESULTS_PATH_)
    app_info.fn_save_model = neural_net_mgr.fn_save_model
    # app_info.fn_save_model = neural_net_mgr.fn_save_model

    training_mgr = training_mgt(game_mgr, neural_net_mgr, app_info)


    @tracer(app_info, verboscity= 4)
    def fn_train():
        training_mgr.fn_execute_training_iterations()
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
        system_nn = model_mgt(game_mgr, app_info.RESULTS_PATH_)
        if not system_nn.fn_load_model():
            return

        system_mcts = monte_carlo_tree_search_mgt(game_mgr, system_nn, app_info)
        fn_system_policy = lambda state: numpy.argmax(system_mcts.fn_get_policy(state, do_random_selection=False))
        fn_contender_policy = fn_player_policy(game_mgr)
        playground = playground_mgt(fn_system_policy, fn_contender_policy, game_mgr,
                                    fn_display=game_mgt(app_info.BOARD_SIZE).fn_display,
                                    )
        system_wins, system_losses, draws = playground.fn_play_games(NUM_TEST_GAMES, verbose=verbose)

        app_info.trace_mgr.fn_write(f'wins:{system_wins} losses:{system_losses} draws:{draws}')

    @tracer(app_info, verboscity= 4)
    def fn_reset():
        if os.path.exists(app_info.RESULTS_PATH_):
            shutil.rmtree(app_info.RESULTS_PATH_)
        return agent_mgr

    agent_mgr = namedtuple('_',
                            [
                               'fn_reset',
                                'fn_train',
                                'fn_test_against_human',
                                'fn_test_againt_random',
                                'fn_test_against_greedy',
                                'fn_change_args',
                                'fn_show_args',
                                'fn_archive_log_file',
                                'app_info'
                           ]
                           )
    agent_mgr.fn_reset = fn_reset
    agent_mgr.fn_train = fn_train
    agent_mgr.fn_test_against_human = fn_test_against_human
    agent_mgr.fn_test_against_random = fn_test_against_random
    agent_mgr.fn_test_against_greedy = fn_test_against_greedy
    agent_mgr.fn_change_args = common_functions.fn_change_args
    agent_mgr.fn_show_args = common_functions.fn_show_args
    agent_mgr.fn_archive_log_file = common_functions.fn_archive_log_file
    agent_mgr.APP_INFO = app_info
    return agent_mgr
