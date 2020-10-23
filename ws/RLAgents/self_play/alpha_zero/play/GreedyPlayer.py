
class GreedyPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, pieces):
        valids = self.game.fn_get_valid_moves(pieces, 1)
        candidates = []
        for a in range(self.game.fn_get_action_size()):
            if valids[a]==0:
                continue
            nextPieces, _ = self.game.fn_get_next_state(pieces, 1, a)
            score = self.game.fn_get_score(nextPieces, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]