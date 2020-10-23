import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.fn_get_action_size())
        valids = self.game.fn_get_valid_moves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.fn_get_action_size())
        return a
