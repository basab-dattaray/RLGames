from collections import deque
from random import shuffle

import numpy as np

from ws.RLAgents.E_SelfPlay.train.training_helper import fn_save_train_examples
from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt
from ws.RLUtils.monitoring.tracing.tracer import tracer


def fn_generate_samples(app_info,
                        game_mgr,
                        iteration,
                        generation_mcts
                        ):
    # game_mgr = app_info.game_mgr
    training_samples_buffer = []

    def _fn_form_sample_data(current_player, run_result, training_samples):
        sample_data = []
        for canon_board, player, canon_policies in training_samples:
            result = run_result * (1 if (player == current_player) else -1)
            sample_data.append([canon_board, canon_policies, result])
        return sample_data

    def _fn_generate_samples_for_an_iteration():

        def _fn_run_one_episode(current_pieces, cur_player_index, episode_step):
            samples_from_episodes = []
            canonical_board_pieces = game_mgr.fn_get_canonical_form(current_pieces, cur_player_index)
            policy = generation_mcts.fn_get_policy(
                canonical_board_pieces, do_random_selection=int(episode_step < app_info.PROBABILITY_SPREAD_THRESHOLD))

            if policy is None:
                return None

            symmetric_samples = game_mgr.fn_get_symmetric_samples(canonical_board_pieces, policy)

            for sym_canon_board, canon_policies in symmetric_samples:
                samples_from_episodes.append((sym_canon_board, cur_player_index, canon_policies))

            action = np.random.choice(len(policy), p=policy)
            next_pieces = game_mgr.fn_get_next_state(current_pieces, cur_player_index, action)
            next_player_index = -1 * cur_player_index

            current_pieces = next_pieces
            return samples_from_episodes, current_pieces, next_player_index

        all_samples_from_iteration = []
        current_pieces = game_mgr.fn_get_init_board()
        curPlayer = 1
        episode_step = 0

        while True:
            episode_step += 1

            samples_from_episodes, current_pieces, curPlayer = \
                _fn_run_one_episode(current_pieces, curPlayer, episode_step)

            all_samples_from_iteration.extend(samples_from_episodes)

            game_status = game_mgr.fn_get_game_progress_status(current_pieces, curPlayer)

            if game_status != 0 or curPlayer is None:
                return _fn_form_sample_data(curPlayer, game_status, all_samples_from_iteration)

    # @tracer(app_info)
    def _fn_generate_samples_for_iteration():
        samples_for_iteration = deque([], maxlen=app_info.SAMPLE_BUFFER_SIZE)
        fn_count_event, fn_stop_counting = progress_count_mgt('Episodes', app_info.NUM_TRAINING_EPISODES)

        for episode_num in range(1, app_info.NUM_TRAINING_EPISODES + 1):
            fn_count_event()
            episode_result = _fn_generate_samples_for_an_iteration()
            if episode_result is not None:
                samples_for_iteration += episode_result

        fn_stop_counting()
        return samples_for_iteration

    all_samples = _fn_generate_samples_for_iteration()
    training_samples_buffer.append(all_samples)

    if len(training_samples_buffer) > app_info.SAMPLE_HISTORY_BUFFER_SIZE:
        app_info.logger.warning(
            f"Removing the oldest entry in training_samples. len(training_samples_buffer) = {len(training_samples_buffer)}")
        training_samples_buffer.pop(0)

    fn_save_train_examples(app_info, iteration - 1, training_samples_buffer)
    training_samples = []
    for samples in training_samples_buffer:
        training_samples.extend(samples)
    shuffle(training_samples)

    return training_samples



