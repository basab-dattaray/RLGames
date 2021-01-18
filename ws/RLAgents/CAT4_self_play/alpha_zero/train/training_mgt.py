import copy
from collections import namedtuple


import numpy as np
from pip._vendor.colorama import Fore

from ws.RLAgents.CAT4_self_play.alpha_zero.play.playground_mgt import playground_mgt
# from ws.RLAgents.CAT4_self_play.alpha_zero.search.mcts_adapter import mcts_adapter
from ws.RLAgents.algo_lib.logic.search.monte_carlo_tree_search_mgt import monte_carlo_tree_search_mgt
from ws.RLAgents.CAT4_self_play.alpha_zero.train.sample_generator import fn_generate_samples
from ws.RLAgents.CAT4_self_play.alpha_zero.train.training_helper import fn_getCheckpointFile, fn_log_iteration_results
from ws.RLUtils.monitoring.tracing.tracer import tracer
def training_mgt(nn_mgr_N, app_info):
    _TMP_MODEL_FILENAME = '_Tmp'
    nn_mgr_P = copy.deepcopy(nn_mgr_N)

    def _fn_try_to_load_model():
        if app_info.DO_LOAD_MODEL:
            if not app_info.neural_net_mgr.fn_load_model():
                app_info.fn_log('*** unable to load model')
            else:
                app_info.fn_log('!!! loaded model')


    @tracer(app_info)
    def fn_execute_training_iterations():
        game_mgr = app_info.game_mgr

        update_count = 0
        @tracer(app_info)

        def _fn_interpret_competition_results(iteration, nwins, pwins):
            # def _fn_write_result(color, result, update_score, do_save):
            #     app_info.fn_log(
            #         color + result + ' New Model: update_threshold: {}, update_score: {}'.format(
            #             app_info.SCORE_BASED_MODEL_UPDATE_THRESHOLD,
            #             update_score))
            #     if do_save:
            #         nn_mgr_N.fn_save_model(model_file_name= fn_getCheckpointFile(iteration))
            #         nn_mgr_N.fn_save_model()
            #     else:
            #         nn_mgr_N.fn_load_model(model_file_name= app_info.RESULTS_PATH_)

            nonlocal update_count
            reject = False
            update_score = 0
            if pwins + nwins == 0:
                reject = True
            else:
                update_score = float(nwins) / (pwins + nwins)
                if update_score < app_info.SCORE_BASED_MODEL_UPDATE_THRESHOLD:
                    reject = True
            model_already_exists = nn_mgr_N.fn_is_model_available(app_info.RESULTS_PATH_) ###

            if not reject:
                update_count += 1

            if reject and not model_already_exists:
                app_info.fn_log(f'MODEL ACCEPTED: score: {update_score} pass: {app_info.SCORE_BASED_MODEL_UPDATE_THRESHOLD}')
            else:
                if reject:
                    app_info.fn_log(
                        f'MODEL REJECTED: score: {update_score} pass: {app_info.SCORE_BASED_MODEL_UPDATE_THRESHOLD}')
                else:
                    app_info.fn_log(
                        f'MODEL ACCEPTED: score: {update_score} pass: {app_info.SCORE_BASED_MODEL_UPDATE_THRESHOLD}')

            # app_info.fn_log('')

        def fn_run_iteration(iteration):
            nonlocal update_count
            app_info.fn_log('')
            app_info.fn_log(f'ITERATION NUMBER {iteration} of {app_info.NUM_TRAINING_ITERATIONS}')

            @tracer(app_info)
            def _fn_play_next_vs_previous(training_samples):
                nn_mgr_N.fn_save_model(_TMP_MODEL_FILENAME)
                nn_mgr_P.fn_load_model(_TMP_MODEL_FILENAME)
                pmcts = monte_carlo_tree_search_mgt(app_info.game_mgr, nn_mgr_P, app_info)
                nn_mgr_N.fn_adjust_model_from_examples(training_samples, app_info.NUM_EPOCHS)
                nmcts = monte_carlo_tree_search_mgt(app_info.game_mgr, nn_mgr_N, app_info)
                playground = playground_mgt(
                    lambda state: np.argmax(pmcts.fn_get_policy(state, do_random_selection= False)),
                    lambda state: np.argmax(nmcts.fn_get_policy(state, do_random_selection= False)),
                    game_mgr
                )
                pwins, nwins, draws = playground.fn_play_games(app_info.NUM_GAMES_FOR_MODEL_COMPARISON)
                app_info.fn_log()
                return draws, nwins, pwins

            training_samples = fn_generate_samples(app_info, iteration,  monte_carlo_tree_search_mgt(app_info.game_mgr, nn_mgr_N, app_info))
            draws, nwins, pwins = _fn_play_next_vs_previous(training_samples)
            fn_log_iteration_results(app_info, draws, iteration, nwins, pwins)

            _fn_interpret_competition_results(iteration, nwins, pwins)

        _fn_try_to_load_model()
        for iteration in range(1, app_info.NUM_TRAINING_ITERATIONS + 1):
            fn_run_iteration(iteration)
            if update_count >= app_info.NUM_OF_ITERATION_SUCCESSES_FOR_MODEL_UPGRADE:
                break

    training_mgr  = namedtuple('_', ['fn_execute_training_iterations'])
    training_mgr.fn_execute_training_iterations = fn_execute_training_iterations

    return training_mgr