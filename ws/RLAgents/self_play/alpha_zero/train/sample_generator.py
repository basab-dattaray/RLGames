from collections import deque
from random import shuffle

import numpy as np

from ws.RLAgents.self_play.alpha_zero.train.training_helper import fn_save_train_examples
from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt
from ws.RLUtils.monitoring.tracing.tracer import tracer


def fn_generate_samples(args, iteration, generation_mcts):
    game_mgr = args.game_mgr
    training_samples_buffer = []

    def _fn_form_sample_data(current_player, run_result, training_samples):
        sample_data = []
        for canon_board, player, canon_action_probs in training_samples:
            result = run_result * (1 if (player == current_player) else -1)
            sample_data.append([canon_board, canon_action_probs, result])
        return sample_data

    def _fn_generate_samples_for_an_iteration():
        # trainExamples = []
        all_samples_from_iteration = []
        current_pieces = game_mgr.fn_get_init_board()
        curPlayer = 1
        episode_step = 0

        def _fn_run_one_episode(current_pieces, curPlayer, episode_step):
            samples_from_episodes = []
            canonical_board_pieces = game_mgr.fn_get_canonical_form(current_pieces, curPlayer)
            spread_probabilities = int(episode_step < args.probability_spread_threshold)

            action_probs = generation_mcts.fn_get_policy\
                (canonical_board_pieces, spread_probabilities=spread_probabilities)

            if action_probs is None:
                return None

            symmetric_samples = game_mgr.fn_get_symetric_samples(canonical_board_pieces, action_probs)
            # samples = map(lambda b, p: samples.append([b, curPlayer, p, None]), sym)
            for sym_canon_board, canon_action_probs in symmetric_samples:
                # samples.append((sym_canon_board, curPlayer, canon_action_probs))
                samples_from_episodes.append((sym_canon_board, curPlayer, canon_action_probs))

            action = np.random.choice(len(action_probs), p=action_probs)
            next_pieces, player_next = game_mgr.fn_get_next_state(current_pieces, curPlayer, action)

            current_pieces = next_pieces
            return samples_from_episodes, current_pieces, player_next

        while True:
            episode_step += 1

            samples_from_episodes, current_pieces, curPlayer = \
                _fn_run_one_episode(current_pieces, curPlayer, episode_step)

            all_samples_from_iteration.extend(samples_from_episodes)

            game_status = game_mgr.fn_get_game_progress_status(current_pieces, curPlayer)

            if game_status != 0 or curPlayer is None:
                return _fn_form_sample_data(curPlayer, game_status, all_samples_from_iteration)

    @tracer(args)
    def _fn_generate_all_samples(training_samples_buffer):
            samples_for_iteration = deque([], maxlen=args.sample_buffer_size)
            fn_count_event, fn_stop_counting = progress_count_mgt('Episodes', args.num_of_training_episodes)

            for episode_num in range(1, args.num_of_training_episodes + 1):
                fn_count_event()
                episode_result = _fn_generate_samples_for_an_iteration()
                if episode_result is not None:
                    samples_for_iteration += episode_result

            fn_stop_counting()
            args.calltracer.fn_write(f'Number of Episodes for sample generation: {args.num_of_training_episodes}')

            # save the iteration examples to the history
            training_samples_buffer.append(samples_for_iteration)

    _fn_generate_all_samples(training_samples_buffer)

    if len(training_samples_buffer) > args.sample_history_buffer_size:
        args.logger.warning(
            f"Removing the oldest entry in trainExamples. len(training_samples_buffer) = {len(training_samples_buffer)}")
        training_samples_buffer.pop(0)

    fn_save_train_examples(args, iteration - 1, training_samples_buffer)

    training_samples = []
    for samples in training_samples_buffer:
        training_samples.extend(samples)
    shuffle(training_samples)

    return training_samples



