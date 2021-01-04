
def greedy_player_mgt(game_mgr):

    game_mgr = game_mgr

    def fn_get_action(pieces):
        valid_moves = game_mgr.fn_get_valid_moves(pieces, 1)
        if valid_moves is None:
            return None
        candidates = []
        for a in range(game_mgr.fn_get_action_size()):
            if valid_moves[a]==0:
                continue
            nextPieces = game_mgr.fn_get_next_state(pieces, 1, a)
            score = game_mgr.fn_get_score(nextPieces, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

    return fn_get_action