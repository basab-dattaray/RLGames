import functools
import logging

from random import random

from pip._vendor.colorama import Fore

from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt

log = logging.getLogger(__name__)

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, fn_display=None, msg_recorder = None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.fn_display = fn_display
        self.game_num = 0
        self.msg_recorder = msg_recorder

    def playGame(self, verbose=False):

        DEBUG = False

        players = [self.player2, None, self.player1]
        cur_player_index = 1
        pieces = self.game.fn_get_init_board()
        it = 0

        break_from_while = False
        while self.game.fn_get_game_progress_status(pieces, cur_player_index) == 0:
            it += 1
            if verbose:
                assert self.fn_display
                print("Turn ", str(it), "Player ", str(cur_player_index))
                self.fn_display(pieces)

            cur_player = self.player1 if cur_player_index == 1 else self.player2

            # cur_player = players[cur_player_index + 1]
            action = cur_player(self.game.fn_get_canonical_form(pieces, cur_player_index))

            valids = self.game.fn_get_valid_moves(self.game.fn_get_canonical_form(pieces, cur_player_index), 1)

            if valids[action] == 0:
                if DEBUG:
                    x = (int) (action / len(pieces))
                    y = action % len(pieces)
                    self.msg_recorder(f'Action {action} is not valid!   [{x} {y}]')

                    self.msg_recorder(f'Current Player: {cur_player_index} ')


                    self.msg_recorder(f'valids = {valids}')
                    self.msg_recorder('')

                    for i in range(len(pieces)):
                        # arr_of_strs = map(lambda size: '{0:03d}'.format(size), board_pieces[i])
                        # line = list(arr_of_strs)
                        lst = list(map(lambda n: '0' if n == 0 else '+' if n > 0 else '-', pieces[i]))
                        line = functools.reduce(lambda a,b : a + ' ' + b,lst)
                        self.msg_recorder(line)
                break_from_while = True
                break

                # log.debug(f'valids = {valids}')
                # assert valids[action] > 0
            pieces, cur_player_index = self.game.fn_get_next_state(pieces, cur_player_index, action)

        game_status1 = self.game.fn_get_game_progress_status(pieces, cur_player_index)
        game_status = self.game.fn_game_status(pieces)

        if verbose:
            assert self.fn_display
            print("Game over: Turn ", str(it), "Result ", str(game_status))
            self.fn_display(pieces)


        result = game_status
        if DEBUG:
            color = Fore.RED
            result1 = cur_player_index * game_status1
            if result == result1:
                color = Fore.GREEN

            self.msg_recorder(color + f'curPlayer= {cur_player_index}')
            self.msg_recorder(f'RESULT:: {result}')
            self.msg_recorder(f'RESULT1:: {result1}  --> old')

            self.msg_recorder(Fore.BLUE)
        # print(f'Game number={self.game_num}; curPlayer={curPlayer}; result={result}')
        self.game_num += 1
        return result


    def playGames(self, num_of_games, verbose=False):
        self.count_episode, self.end_couunting = progress_count_mgt('Game Counts', num_of_games)
        num_div_2 = int(num_of_games / 2)
        extra_for_1 = 0
        extra_for_2 = 0
        if num_of_games % 2 == 1:
            if random() > .5:
                extra_for_1 = 1
            else:
                extra_for_2 = 1

        oneWon_1, twoWon_1, draws_1 = self.fn_get_gameset_results(num_div_2 + extra_for_1, 1, verbose)
        self.player1, self.player2 = self.player2, self.player1
        oneWon_2, twoWon_2, draws_2 = self.fn_get_gameset_results(num_div_2 + extra_for_2, -1, verbose)

        self.end_couunting()
        return oneWon_1 + oneWon_2, twoWon_1 + twoWon_2, draws_1 + draws_2

    def fn_get_gameset_results(self, num, result_factor, verbose):
        oneWon = 0
        twoWon = 0
        draws = 0
        for i in range(num):
            self.count_episode()
            gameResult = self.playGame(verbose=verbose)

            if gameResult == 1 * result_factor:
                oneWon += 1
            elif gameResult == -1 * result_factor:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws









