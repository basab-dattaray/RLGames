import logging
from random import random

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/RandomPlayer.py for an example. See demo_test_system_vs_human.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.game_num = 0

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """

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
                break
                # log.error(f'Action {action} is not valid!')
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


    def playGames(self, num, verbose=False):

        num = int(num / 2)
        extra_for_1 = 0
        extra_for_2 = 0
        if num % 2 == 1:
            if random() > .5:
                extra_for_1 = 1
            else:
                extra_for_2 = 1

        oneWon_1, twoWon_1, draws_1 = self.fn_get_gameset_results(num + extra_for_1, 1, verbose)
        self.player1, self.player2 = self.player2, self.player1
        oneWon_2, twoWon_2, draws_2 = self.fn_get_gameset_results(num + extra_for_2, -1, verbose)
        return oneWon_1 + oneWon_2, twoWon_1 + twoWon_2, draws_1 + draws_2

    def fn_get_gameset_results(self, num, result_factor, verbose):
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1 * result_factor:
                oneWon += 1
            elif gameResult == -1 * result_factor:
                twoWon += 1
            else:
                draws += 1
        return oneWon, twoWon, draws
