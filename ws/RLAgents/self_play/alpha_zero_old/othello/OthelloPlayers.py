import numpy as np


class fn_get_random_policy():
    def __init__(self, game):
        self.game = game

    def fn_play_it(self, board, player= None):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        if sum(valids) == 0:
            return None

        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())

        return a


class fn_get_human_policy():
    def __init__(self, game):
        self.game = game

    def fn_play_it(self, board, player= None):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        if sum(valid) == 0:
            return None
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def fn_play_it(self, board, player= None):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _, early_completion_result = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
