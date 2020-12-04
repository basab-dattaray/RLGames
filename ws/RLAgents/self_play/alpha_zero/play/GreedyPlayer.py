
class GreedyPlayer():
    def __init__(self, game_mgr):
        self.game_mgr = game_mgr

    def fn_get_action(self, pieces):
        valid_moves = self.game_mgr.fn_get_valid_moves(pieces, 1)
        if valid_moves is None:
            return None
        candidates = []
        for a in range(self.game_mgr.fn_get_action_size()):
            if valid_moves[a]==0:
                continue
            nextPieces, _ = self.game_mgr.fn_get_next_state(pieces, 1, a)
            score = self.game_mgr.fn_get_score(nextPieces, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]