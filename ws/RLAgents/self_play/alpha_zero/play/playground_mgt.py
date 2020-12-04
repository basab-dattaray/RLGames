
from collections import namedtuple

from random import random

from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt


def playground_mgt(fn_policy_player1, fn_policy_player2, game_mgr, fn_display=None, msg_recorder = None):
    game_num = 0
    def _fn_play_game(verbose=False):
        def _fn_switch_policy(cur_player_index):
            curent_policy = fn_policy_player1 if cur_player_index == 1 else fn_policy_player2
            return curent_policy

        def _fn_display_if_verbose(verbose):
            if verbose:
                assert fn_display
                print()
                print("Turn ", str(loop_count), "Player ", str(cur_player_index))
                fn_display(pieces)

        nonlocal game_num

        cur_player_index = 1
        pieces = game_mgr.fn_get_init_board()
        loop_count = 0

        while game_mgr.fn_get_game_progress_status(pieces, cur_player_index) == 0:
            loop_count += 1
            _fn_display_if_verbose(verbose)

            curent_policy = _fn_switch_policy(cur_player_index)

            canonical_pieces = game_mgr.fn_get_canonical_form(pieces, cur_player_index)


            valid_moves = game_mgr.fn_get_valid_moves(canonical_pieces, 1)
            if valid_moves is None:
                break

            action = curent_policy(canonical_pieces)
            if action == None:
                break

            pieces, cur_player_index = game_mgr.fn_get_next_state(pieces, cur_player_index, action)

        game_status = game_mgr.fn_game_status(pieces)

        _fn_display_if_verbose(verbose)

        result = game_status
        game_num += 1
        return result



    def fn_play_games(num_of_games, verbose=False):
        nonlocal fn_policy_player1, fn_policy_player2

        def _fn_get_gameset_results(num, result_factor, verbose):
            oneWon = 0
            twoWon = 0
            draws = 0
            for i in range(num):
                fn_count_event()
                gameResult = _fn_play_game(verbose=verbose)

                if gameResult == 1 * result_factor:
                    oneWon += 1
                elif gameResult == -1 * result_factor:
                    twoWon += 1
                else:
                    draws += 1

            return oneWon, twoWon, draws

        fn_count_event, fn_stop_counting = progress_count_mgt('Game Counts', num_of_games)

        num_div_2 = int(num_of_games / 2)
        extra_for_1 = 0
        extra_for_2 = 0
        if num_of_games % 2 == 1:
            if random() > .5:
                extra_for_1 = 1
            else:
                extra_for_2 = 1

        oneWon_1, twoWon_1, draws_1 = _fn_get_gameset_results(num_div_2 + extra_for_1, 1, verbose)
        fn_policy_player1, fn_policy_player2 = fn_policy_player2, fn_policy_player1
        oneWon_2, twoWon_2, draws_2 = _fn_get_gameset_results(num_div_2 + extra_for_2, -1, verbose)

        fn_stop_counting()
        return oneWon_1 + oneWon_2, twoWon_1 + twoWon_2, draws_1 + draws_2

    playground_mgr = namedtuple('_', ['fn_play_games'])
    playground_mgr.fn_play_games=fn_play_games

    return playground_mgr










