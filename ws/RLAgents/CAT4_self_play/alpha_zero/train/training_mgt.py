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
def training_mgt(nn_mgr_N, args):
    TMP_MODEL_FILENAME_ = '_tmp.tar'
    nn_mgr_P = copy.deepcopy(nn_mgr_N)

    def _fn_try_to_load_model():
        if args.DO_LOAD_MODEL:
            if not args.neural_net_mgr.fn_load_model():
                args.fn_record('*** unable to load model')
            else:
                args.fn_record('!!! loaded model')


    @tracer(args)
    def fn_execute_training_iterations():
        game_mgr = args.game_mgr

        update_count = 0
        @tracer(args)

        def _fn_interpret_competition_results(iteration, nwins, pwins):
            def _fn_write_result(color, result, update_score, do_save):
                args.CALL_TRACER_.fn_write(
                    color + result + ' New Model: update_threshold: {}, update_score: {}'.format(
                        args.SCORE_BASED_MODEL_UPDATE_THRESHOLD,
                        update_score))
                if do_save:
                    nn_mgr_N.fn_save_model(filename=fn_getCheckpointFile(iteration))
                    nn_mgr_N.fn_save_model()
                else:
                    nn_mgr_N.fn_load_model(filename=TMP_MODEL_FILENAME_)

            nonlocal update_count
            reject = False
            update_score = 0
            if pwins + nwins == 0:
                reject = True
            else:
                update_score = float(nwins) / (pwins + nwins)
                if update_score < args.SCORE_BASED_MODEL_UPDATE_THRESHOLD:
                    reject = True
            model_already_exists = nn_mgr_N.fn_is_model_available(args.RESULTS_PATH_)

            if not reject:
                update_count += 1

            if reject and not model_already_exists:
                _fn_write_result(Fore.MAGENTA, 'ACCEPTED', update_score, do_save= True)
            else:
                if reject:
                    _fn_write_result(Fore.RED, 'REJECTED', update_score, do_save= False)
                else:
                    _fn_write_result(Fore.GREEN, 'ACCEPTED', update_score, do_save= True)

            args.CALL_TRACER_.fn_write(Fore.BLACK)

        def fn_run_iteration(iteration):
            nonlocal update_count
            args.CALL_TRACER_.fn_write('')
            args.CALL_TRACER_.fn_write(f'ITERATION NUMBER {iteration} of {args.NUM_TRAINING_ITERATIONS}', indent=0)

            @tracer(args)
            def _fn_play_next_vs_previous(training_samples):
                nn_mgr_N.fn_save_model(filename=TMP_MODEL_FILENAME_)
                nn_mgr_P.fn_load_model(filename=TMP_MODEL_FILENAME_)
                pmcts = monte_carlo_tree_search_mgt(args.game_mgr, nn_mgr_P, args)
                nn_mgr_N.fn_adjust_model_from_examples(training_samples, args.NUM_EPOCHS)
                nmcts = monte_carlo_tree_search_mgt(args.game_mgr, nn_mgr_N, args)
                playground = playground_mgt(
                    lambda state: np.argmax(pmcts.fn_get_policy(state, do_random_selection= False)),
                    lambda state: np.argmax(nmcts.fn_get_policy(state, do_random_selection= False)),
                    game_mgr,
                    msg_recorder=args.CALL_TRACER_.fn_write
                )
                pwins, nwins, draws = playground.fn_play_games(args.NUM_GAMES_FOR_MODEL_COMPARISON)
                args.fn_record()
                return draws, nwins, pwins

            training_samples = fn_generate_samples(args, iteration,  monte_carlo_tree_search_mgt(args.game_mgr, nn_mgr_N, args))
            draws, nwins, pwins = _fn_play_next_vs_previous(training_samples)
            fn_log_iteration_results(args, draws, iteration, nwins, pwins)

            _fn_interpret_competition_results(iteration, nwins, pwins)

        _fn_try_to_load_model()
        for iteration in range(1, args.NUM_TRAINING_ITERATIONS + 1):
            fn_run_iteration(iteration)
            if update_count >= args.NUM_OF_ITERATION_SUCCESSES_FOR_MODEL_UPGRADE:
                break

    training_mgr  = namedtuple('_', ['fn_execute_training_iterations'])
    training_mgr.fn_execute_training_iterations = fn_execute_training_iterations

    return training_mgr