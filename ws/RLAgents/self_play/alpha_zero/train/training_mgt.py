import copy
import logging
import os
import sys
from collections import deque, namedtuple
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from pip._vendor.colorama import Fore

from ws.RLAgents.self_play.alpha_zero.play.Arena import Arena
from ws.RLAgents.self_play.alpha_zero.search.mcts_adapter import mcts_adapter
# from ws.RLAgents.self_play.alpha_zero.search.recursive.MCTS import MCTS
from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt
from ws.RLUtils.monitoring.tracing.tracer import tracer

log = logging.getLogger(__name__)

def training_mgt(game, nnet, args):

    DEBUG_FLAG = False

    pnet = copy.deepcopy(nnet)

    def _fn_getCheckpointFile(iteration):
        return 'model_' + str(iteration) + '.tar'

    def _fn_save_train_examples(iteration):
        folder = args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, _fn_getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(training_samples_buffer)
        # f.closed

    def _fn_load_train_examples():
        modelFile = os.path.join(args.load_folder_file[0], args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|size]")
            if r != "y":
                sys.exit()
        else:
            args.fn_record("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                training_samples_buffer = Unpickler(f).load()
            args.fn_record('Loading done!')

            # examples based on the model were already collected (loaded)
            skipFirstSelfPlay = True

    def _fn_record_iter_results(draws, iteration, nwins, pwins):
        args.recorder.fn_record_message(f'-- Iter {iteration} of {args.numIters}', indent=0)
        update_threshold = 'update threshold: {}'.format(args.score_based_model_update_threshold)
        args.recorder.fn_record_message(update_threshold)
        score = f'nwins:{nwins} pwins:{pwins} draws:{draws}'
        args.recorder.fn_record_message(score)

    training_samples_buffer = []  # history of examples from args.sample_history_buffer_size latest iterations
    skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def fn_form_sample_data(current_player, run_result, training_samples):
        sample_data = []
        for canon_board, player, canon_action_probs in training_samples:
            result = run_result * (1 if (player == current_player) else -1)
            sample_data.append([canon_board, canon_action_probs, result])
        return sample_data

    @tracer(args)
    def fn_execute_training_iterations():

        @tracer(args)
        def fn_run_iteration(iteration):

            @tracer(args)
            def fn_generate_samples(iteration):
                generation_mcts = mcts_adapter(game, nnet, args)

                def _fn_run_episodes():
                    trainExamples = []
                    current_pieces = game.fn_get_init_board()
                    curPlayer = 1
                    episode_step = 0

                    while True:
                        episode_step += 1
                        canonical_board_pieces = game.fn_get_canonical_form(current_pieces, curPlayer)
                        spread_probabilities = int(episode_step < args.probability_spread_threshold)

                        action_probs = generation_mcts.fn_get_action_probabilities(canonical_board_pieces,
                                                                                   spread_probabilities=spread_probabilities)
                        if action_probs is None:
                            return None

                        symetric_samples = game.fn_get_symetric_samples(canonical_board_pieces, action_probs)
                        # trainExamples = map(lambda b, p: trainExamples.append([b, curPlayer, p, None]), sym)
                        for sym_canon_board, canon_action_probs in symetric_samples:
                            trainExamples.append((sym_canon_board, curPlayer, canon_action_probs))

                        action = np.random.choice(len(action_probs), p=action_probs)
                        next_pieces, player_next = game.fn_get_next_state(current_pieces, curPlayer, action)

                        if DEBUG_FLAG:
                            print()
                            print('player:{}'.format(curPlayer))
                            print()
                            print(next_pieces)

                        curPlayer = player_next
                        current_pieces = next_pieces

                        result = game.fn_get_game_progress_status(current_pieces, curPlayer)

                        if result != 0 or player_next is None:
                            return fn_form_sample_data(curPlayer, result, trainExamples)

                # examples of the iteration
                if not skipFirstSelfPlay or iteration > 1:
                    samples_for_iteration = deque([], maxlen=args.sample_buffer_size)
                    fn_count_episode, fn_end_couunting = progress_count_mgt('Episodes', args.num_of_training_episodes)
                    for episode_num in range(1, args.num_of_training_episodes + 1):
                        fn_count_episode()

                        # mcts = mcts_adapter(game, nnet, args)  # reset search tree
                        episode_result = _fn_run_episodes()
                        if episode_result is not None:
                            samples_for_iteration += episode_result
                    fn_end_couunting()
                    args.recorder.fn_record_message(f'Number of Episodes for sample generation: {args.num_of_training_episodes}')

                    # save the iteration examples to the history
                    training_samples_buffer.append(samples_for_iteration)
                if len(training_samples_buffer) > args.sample_history_buffer_size:
                    log.warning(
                        f"Removing the oldest entry in trainExamples. len(training_samples_buffer) = {len(training_samples_buffer)}")
                    training_samples_buffer.pop(0)
                # backup history to a file
                # NB! the examples were collected using the model from the previous iteration, so (iteration-1)
                _fn_save_train_examples(iteration - 1)
                # shuffle examples before training
                trainExamples = []
                for e in training_samples_buffer:
                    trainExamples.extend(e)
                shuffle(trainExamples)

                return trainExamples


            @tracer(args)
            def _fn_play_next_vs_previous(trainExamples):
                # training new network, keeping a copy of the old one
                nnet.save_checkpoint(rel_folder=args.checkpoint, filename='temp.tar')
                pnet.load_checkpoint(rel_folder=args.checkpoint, filename='temp.tar')
                pmcts = mcts_adapter(game, pnet, args)
                nnet.fn_adjust_model_from_examples(trainExamples)
                nmcts = mcts_adapter(game, nnet, args)
                # args.recorder.fn_record_message()
                # args.recorder.fn_record_message(f'* Comptete with Previous Version', indent=0)
                arena = Arena(lambda x: np.argmax(pmcts.fn_get_action_probabilities(x, spread_probabilities=0)),
                              lambda x: np.argmax(nmcts.fn_get_action_probabilities(x, spread_probabilities=0)),
                              game,
                              msg_recorder=args.recorder.fn_record_message)
                pwins, nwins, draws = arena.play_games(args.number_of_games_for_model_comarison)
                args.fn_record()
                return draws, nwins, pwins

            trainExamples = fn_generate_samples(iteration)

            draws, nwins, pwins = _fn_play_next_vs_previous(trainExamples)

            _fn_record_iter_results(draws, iteration, nwins, pwins)

            reject = False
            update_score = 0
            if pwins + nwins == 0:
                reject = True
            else:
                update_score = float(nwins) / (pwins + nwins)
                if update_score < args.score_based_model_update_threshold:
                    reject = True

            model_already_exists = nnet.fn_is_model_available(rel_folder=args.checkpoint)

            if reject and model_already_exists:
                color = Fore.RED
                args.recorder.fn_record_message(
                    color + 'REJECTED New Model: update_threshold: {}, update_score: {}'.format(args.score_based_model_update_threshold,
                                                                                                update_score))
                nnet.load_checkpoint(rel_folder=args.checkpoint, filename='temp.tar')
            else:
                color = Fore.GREEN
                args.recorder.fn_record_message(
                    color + 'ACCEPTED New Model: update_threshold: {}, update_score: {}'.format(args.score_based_model_update_threshold,
                                                                                                update_score))
                nnet.save_checkpoint(rel_folder=args.checkpoint, filename=_fn_getCheckpointFile(iteration))
                nnet.save_checkpoint(rel_folder=args.checkpoint, filename='model.tar')
            args.recorder.fn_record_message(Fore.BLACK)



        if args.load_model:
            # args.fn_record("  Loading 'trainExamples' from file...")
            _fn_load_train_examples()
        for iteration in range(1, args.numIters + 1):
            fn_run_iteration(iteration)



    return fn_execute_training_iterations