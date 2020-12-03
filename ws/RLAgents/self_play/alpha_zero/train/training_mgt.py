import copy
from collections import namedtuple


import numpy as np
from pip._vendor.colorama import Fore

from ws.RLAgents.self_play.alpha_zero.play.playground_mgt import playground_mgt
from ws.RLAgents.self_play.alpha_zero.search.mcts_adapter import mcts_adapter
from ws.RLAgents.self_play.alpha_zero.train.sample_generator import fn_generate_samples
from ws.RLAgents.self_play.alpha_zero.train.training_helper import  fn_log_iteration_results, fn_getCheckpointFile

from ws.RLUtils.monitoring.tracing.tracer import tracer
def training_mgt(nn_mgr_N, args):
    nn_mgr_P = copy.deepcopy(nn_mgr_N)

    @tracer(args)
    def fn_execute_training_iterations():
        game_mgr = args.game_mgr

        update_count = 0
        @tracer(args)

        def _fn_pitting_results(iteration, nwins, pwins):
            nonlocal update_count
            reject = False
            update_score = 0
            if pwins + nwins == 0:
                reject = True
            else:
                update_score = float(nwins) / (pwins + nwins)
                if update_score < args.score_based_model_update_threshold:
                    reject = True
            model_already_exists = nn_mgr_N.fn_is_model_available(rel_folder=args.rel_model_path)
            if reject and model_already_exists:
                color = Fore.RED
                args.calltracer.fn_write(
                    color + 'REJECTED New Model: update_threshold: {}, update_score: {}'.format(
                        args.score_based_model_update_threshold,
                        update_score))
                nn_mgr_N.fn_load_model(filename=args.temp_model_exchange_filename)
            else:
                color = Fore.GREEN
                if reject:  # implies that there is no model on disk yet, hence ACCEPT unacceptable update score for now
                    color = Fore.MAGENTA
                args.calltracer.fn_write(
                    color + 'ACCEPTED New Model: update_threshold: {}, update_score: {}'.format(
                        args.score_based_model_update_threshold,
                        update_score))
                nn_mgr_N.fn_save_model(filename=fn_getCheckpointFile(iteration))
                nn_mgr_N.fn_save_model()
            if not reject:  # continue update if the GREEN color is forced
                update_count += 1
            args.calltracer.fn_write(Fore.BLACK)

        def fn_run_iteration(iteration):
            nonlocal update_count

            @tracer(args)
            def _fn_play_next_vs_previous(training_samples):
                nn_mgr_N.fn_save_model(filename=args.temp_model_exchange_filename)
                nn_mgr_P.fn_load_model(filename=args.temp_model_exchange_filename)
                pmcts = mcts_adapter(nn_mgr_P, args)
                nn_mgr_N.fn_adjust_model_from_examples(training_samples)
                nmcts = mcts_adapter(nn_mgr_N, args)
                arena = playground_mgt(
                    lambda x: np.argmax(pmcts.fn_get_policy(x, do_random_selection= False)),
                    lambda x: np.argmax(nmcts.fn_get_policy(x, do_random_selection= False)),
                    game_mgr,
                    msg_recorder=args.calltracer.fn_write
                )
                pwins, nwins, draws = arena.fn_play_games(args.number_of_games_for_model_comarison)
                args.fn_record()
                return draws, nwins, pwins

            training_samples = fn_generate_samples(args, iteration,  mcts_adapter(nn_mgr_N, args))
            draws, nwins, pwins = _fn_play_next_vs_previous(training_samples)
            fn_log_iteration_results(args, draws, iteration, nwins, pwins)

            _fn_pitting_results(iteration, nwins, pwins)

        for iteration in range(1, args.num_of_training_iterations + 1):
            fn_run_iteration(iteration)
            if update_count >= args.num_of_successes_for_model_upgrade:
                break

    training_mgr  = namedtuple('_', ['fn_execute_training_iterations'])
    training_mgr.fn_execute_training_iterations = fn_execute_training_iterations

    return training_mgr