import functools
import logging
from collections import namedtuple

from random import random

from pip._vendor.colorama import Fore

from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt

log = logging.getLogger(__name__)

def Arena(player1, player2, game, fn_display=None, msg_recorder = None):


    game_num = 0

    def _fn_play_game(verbose=False):
        nonlocal game_num
        DEBUG = False

        players = [player2, None, player1]
        cur_player_index = 1
        pieces = game.fn_get_init_board()
        it = 0

        break_from_while = False
        while game.fn_get_game_progress_status(pieces, cur_player_index) == 0:
            it += 1
            if verbose:
                assert fn_display
                print("Turn ", str(it), "Player ", str(cur_player_index))
                fn_display(pieces)

            cur_player = player1 if cur_player_index == 1 else player2

            # cur_player = players[cur_player_index + 1]
            action = cur_player(game.fn_get_canonical_form(pieces, cur_player_index))

            valids = game.fn_get_valid_moves(game.fn_get_canonical_form(pieces, cur_player_index), 1)

            if valids[action] == 0:
                if DEBUG:
                    x = (int) (action / len(pieces))
                    y = action % len(pieces)
                    msg_recorder(f'Action {action} is not valid!   [{x} {y}]')

                    msg_recorder(f'Current Player: {cur_player_index} ')


                    msg_recorder(f'valids = {valids}')
                    msg_recorder('')

                    for i in range(len(pieces)):
                        # arr_of_strs = map(lambda size: '{0:03d}'.format(size), board_pieces[i])
                        # line = list(arr_of_strs)
                        lst = list(map(lambda n: '0' if n == 0 else '+' if n > 0 else '-', pieces[i]))
                        line = functools.reduce(lambda a,b : a + ' ' + b,lst)
                        msg_recorder(line)
                break_from_while = True
                break

                # log.debug(f'valids = {valids}')
                # assert valids[action] > 0
            pieces, cur_player_index = game.fn_get_next_state(pieces, cur_player_index, action)

        game_status = game.fn_game_status(pieces)

        if verbose:
            assert fn_display
            print("Game over: Turn ", str(it), "Result ", str(game_status))
            fn_display(pieces)


        result = game_status
        if DEBUG:
            color = Fore.RED
            game_status1 = game.fn_get_game_progress_status(pieces, cur_player_index)

            result1 = cur_player_index * game_status1
            if result == result1:
                color = Fore.GREEN

            msg_recorder(color + f'curPlayer= {cur_player_index}')
            msg_recorder(f'RESULT:: {result}')
            msg_recorder(f'RESULT1:: {result1}  --> old')

            msg_recorder(Fore.BLUE)
        # print(f'Game number={game_num}; curPlayer={curPlayer}; result={result}')
        game_num += 1
        return result


    def fn_play_games(num_of_games, verbose=False):
        def _fn_get_gameset_results(num, result_factor, verbose):
            oneWon = 0
            twoWon = 0
            draws = 0
            for i in range(num):
                count_episode()
                gameResult = _fn_play_game(verbose=verbose)

                if gameResult == 1 * result_factor:
                    oneWon += 1
                elif gameResult == -1 * result_factor:
                    twoWon += 1
                else:
                    draws += 1

            return oneWon, twoWon, draws

        count_episode, end_couunting = progress_count_mgt('Game Counts', num_of_games)

        nonlocal player1, player2

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

        end_couunting()
        return oneWon_1 + oneWon_2, twoWon_1 + twoWon_2, draws_1 + draws_2

    playground_mgr = namedtuple('_', ['fn_play_games'])
    playground_mgr.fn_play_games=fn_play_games

    return playground_mgr










