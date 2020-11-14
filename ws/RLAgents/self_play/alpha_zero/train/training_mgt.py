import copy
import logging
import os
import sys
from collections import deque, namedtuple
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from pip._vendor.colorama import Fore

from ws.RLAgents.self_play.alpha_zero.play.playground_mgt import playground_mgt
from ws.RLAgents.self_play.alpha_zero.search.mcts_adapter import mcts_adapter
from ws.RLAgents.self_play.alpha_zero.train.sample_generator import fn_generate_samples
from ws.RLAgents.self_play.alpha_zero.train.training_helper import fn_save_train_examples, fn_log_iter_results, \
    fn_getCheckpointFile

from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt
from ws.RLUtils.monitoring.tracing.tracer import tracer

# log = logging.getLogger(__name__)

def training_mgt(nn_mgr_N, args):

    DEBUG_FLAG = False



    nn_mgr_P = copy.deepcopy(nn_mgr_N)



    # training_samples_buffer = []  # history of examples from nn_args.sample_history_buffer_size latest iterations
    # skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()



    @tracer(args)
    def fn_execute_training_iterations():
        game_mgr = args.game_mgr

        update_count = 0
        @tracer(args)
        def fn_run_iteration(iteration):
            nonlocal update_count

            @tracer(args)
            def _fn_play_next_vs_previous(trainExamples):
                nn_mgr_N.fn_save_model(filename=args.temp_model_exchange_name)
                nn_mgr_P.fn_load_model(filename=args.temp_model_exchange_name)
                pmcts = mcts_adapter(nn_mgr_P, args)
                nn_mgr_N.fn_adjust_model_from_examples(trainExamples)
                nmcts = mcts_adapter(nn_mgr_N, args)
                # nn_args.calltracer.fn_write()
                # nn_args.calltracer.fn_write(f'* Comptete with Previous Version', indent=0)
                arena = playground_mgt(lambda x: np.argmax(pmcts.fn_get_action_probabilities(x, spread_probabilities=0)),
                              lambda x: np.argmax(nmcts.fn_get_action_probabilities(x, spread_probabilities=0)),
                              game_mgr,
                              msg_recorder=args.calltracer.fn_write)
                pwins, nwins, draws = arena.fn_play_games(args.number_of_games_for_model_comarison)
                args.fn_record()
                return draws, nwins, pwins

            trainExamples = fn_generate_samples(args, iteration,  mcts_adapter(nn_mgr_N, args))

            draws, nwins, pwins = _fn_play_next_vs_previous(trainExamples)

            fn_log_iter_results(args, draws, iteration, nwins, pwins)

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
                    color + 'REJECTED New Model: update_threshold: {}, update_score: {}'.format(args.score_based_model_update_threshold,
                                                                                                update_score))
                nn_mgr_N.fn_load_model(filename=args.temp_model_exchange_name)
            else:
                color = Fore.GREEN
                args.calltracer.fn_write(
                    color + 'ACCEPTED New Model: update_threshold: {}, update_score: {}'.format(args.score_based_model_update_threshold,
                                                                                                update_score))
                nn_mgr_N.fn_save_model(filename=fn_getCheckpointFile(iteration))
                nn_mgr_N.fn_save_model()
            if not reject: # continue update if the GREEN color is forced
                update_count += 1

            args.calltracer.fn_write(Fore.BLACK)
        #
        # if args.do_load_samples:
        #     args.fn_record("!!!  loading 'samples' from file...")
        #     fn_load_train_examples()

        for iteration in range(1, args.num_of_training_iterations + 1):
            fn_run_iteration(iteration)
            if update_count >= args.num_of_successes_for_model_upgrade:
                break

    training_mgr  = namedtuple('_', ['fn_execute_training_iterations'])
    training_mgr.fn_execute_training_iterations = fn_execute_training_iterations

    return training_mgr