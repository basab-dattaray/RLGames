import numpy as np


class RandomPlayer():
    def __init__(self, game_mgr):
        self.game_mgr = game_mgr

    def play(self, board):
        a = np.random.randint(self.game_mgr.fn_get_action_size())
        valid_moves = self.game_mgr.fn_get_valid_moves(board, 1)
        while valid_moves[a]!=1:
            a = np.random.randint(self.game_mgr.fn_get_action_size())
        return a
