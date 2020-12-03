import functools
import logging
from collections import namedtuple

from random import random

from pip._vendor.colorama import Fore

from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt

# log = logging.getLogger(__name__)

def playground_mgt(player1, player2, game_mgr, fn_display=None, msg_recorder = None):
    game_num = 0

    def _fn_play_game(verbose=False):
        nonlocal game_num
        DEBUG = False

        cur_player_index = 1
        pieces = game_mgr.fn_get_init_board()
        loop_count = 0

        while game_mgr.fn_get_game_progress_status(pieces, cur_player_index) == 0:
            loop_count += 1
            if verbose:
                assert fn_display
                print()
                print("Turn ", str(loop_count), "Player ", str(cur_player_index))
                fn_display(pieces)

            cur_player = player1 if cur_player_index == 1 else player2

            # cur_player = players[cur_player_index + 1]
            action = cur_player(game_mgr.fn_get_canonical_form(pieces, cur_player_index))
            if action == None:
                break
            valid_moves = game_mgr.fn_get_valid_moves(game_mgr.fn_get_canonical_form(pieces, cur_player_index), 1)
            if valid_moves is None:
                break
            if valid_moves[action] == 0:
                 break

            pieces, cur_player_index = game_mgr.fn_get_next_state(pieces, cur_player_index, action)

        game_status = game_mgr.fn_game_status(pieces)

        if verbose:
            assert fn_display
            fn_display(pieces)
            print()
            print("GAME OVER: Turn ", str(loop_count), "Result ", str(game_status))

        result = game_status
        game_num += 1
        return result


    def fn_play_games(num_of_games, verbose=False):
        nonlocal player1, player2

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
        player1, player2 = player2, player1
        oneWon_2, twoWon_2, draws_2 = _fn_get_gameset_results(num_div_2 + extra_for_2, -1, verbose)

        fn_stop_counting()
        return oneWon_1 + oneWon_2, twoWon_1 + twoWon_2, draws_1 + draws_2

    playground_mgr = namedtuple('_', ['fn_play_games'])
    playground_mgr.fn_play_games=fn_play_games

    return playground_mgr










