import logging

from random import random

from ws.RLUtils.monitoring.tracing.progress_count_mgt import progress_count_mgt

log = logging.getLogger(__name__)

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.game_num = 0

    def playGame(self, verbose=False):

        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                break

                # log.debug(f'valids = {valids}')
                # assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        game_status = self.game.getGameEnded(board, curPlayer)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(game_status))
            self.display(board)

        result = curPlayer * game_status
        #! print(f'Game number={self.game_num}; curPlayer={curPlayer}; result={result}')
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









