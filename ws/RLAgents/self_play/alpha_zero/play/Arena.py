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

    def __init__(self, player1, player2, game, display=None, msg_recorder = None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.game_num = 0
        self.msg_recorder = msg_recorder

    def playGame(self, verbose=False):

        DEBUG = True

        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        break_from_while = False
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            # if DEBUG and self.display is not None:
            #     assert self.display
            #     print("Turn ", str(it), "Player ", str(curPlayer))
            #     self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                if DEBUG:
                    x = (int) (action / len(board))
                    y = action % len(board)
                    self.msg_recorder(f'Action {action} is not valid!   [{x} {y}]')

                    self.msg_recorder(f'Current Player: {curPlayer} ')


                    self.msg_recorder(f'valids = {valids}')
                    self.msg_recorder('')

                    for i in range(len(board)):
                        # arr_of_strs = map(lambda n: '{0:03d}'.format(n), board[i])
                        # line = list(arr_of_strs)
                        lst = list(map(lambda n: '0' if n == 0 else '+' if n > 0 else '-', board[i]))
                        line = functools.reduce(lambda a,b : a + ' ' + b,lst)
                        self.msg_recorder(line)
                break_from_while = True
                break

                # log.debug(f'valids = {valids}')
                # assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        game_status1 = self.game.getGameEnded(board, curPlayer)
        game_status = self.game.fn_game_status(board)

        # if valids[action] == 0 and DEBUG and self.display is not None:
        #     assert self.display
        #     print("Game over: Turn ", str(it), "Result ", str(game_status))
        #     self.display(board)

        result1 = curPlayer * game_status1
        result = game_status
        if DEBUG:
            color = Fore.RED
            if result == result1:
                color = Fore.GREEN

            self.msg_recorder(color + f'curPlayer= {curPlayer}')
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









