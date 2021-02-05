import copy
from collections import namedtuple
import numpy as np

from ws.RLAgents.E_SelfPlay.play.playground_mgt import playground_mgt

from ws.RLUtils.algo_lib.search.monte_carlo_tree_search_mgt import monte_carlo_tree_search_mgt
from ws.RLAgents.E_SelfPlay.train.sample_generator import fn_generate_samples
from ws.RLAgents.E_SelfPlay.train.training_helper import fn_log_iteration_results
from ws.RLUtils.monitoring.tracing.tracer import tracer
def training_mgt(nn_mgr_N, app_info):
    _TMP_MODEL_FILENAME = '_tmp'
    nn_mgr_P = copy.deepcopy(nn_mgr_N)

    def _fn_try_to_load_model():
        if app_info.DO_LOAD_MODEL:
            if not nn_mgr_N.fn_load_model():
                app_info.fn_log('*** unable to load model')
            else:
                app_info.fn_log('!!! loaded model')


    @tracer(app_info)
    def fn_execute_training_iterations():
        game_mgr = app_info.game_mgr

        update_count = 0
        @tracer(app_info)

        def _fn_interpret_competition_results(iteration, nwins, pwins):
            nonlocal update_count
            reject = False
            update_score = 0
            if pwins + nwins == 0:
                reject = True
            else:
                update_score = float(nwins) / (pwins + nwins)
                if update_score < app_info.PASSING_SCORE:
                    reject = True
            model_already_exists = nn_mgr_N.fn_is_model_available(app_info.RESULTS_PATH_) ###

            if not reject:
                update_count += 1

            if reject and not model_already_exists:
                app_info.fn_log(f'MODEL CREATED: score: {update_score} pass: {app_info.PASSING_SCORE}')
                nn_mgr_N.fn_save_model()
            else:
                if reject:
                    app_info.fn_log(
                        f'MODEL REJECTED: score: {update_score} pass: {app_info.PASSING_SCORE}')
                else:
                    app_info.fn_log(
                        f'MODEL ACCEPTED: score: {update_score} pass: {app_info.PASSING_SCORE}')
                    nn_mgr_N.fn_save_model()

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

            training_samples = fn_generate_samples(app_info,
                                                   iteration,
                                                   generation_mcts=monte_carlo_tree_search_mgt(app_info.game_mgr, nn_mgr_N, app_info)
                                                   )
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