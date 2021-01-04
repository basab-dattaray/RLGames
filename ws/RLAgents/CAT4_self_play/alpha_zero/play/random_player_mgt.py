import numpy as np

def random_player_mgt(game_mgr):

    def fn_get_action(board):
        action = np.random.randint(game_mgr.fn_get_action_size())
        valid_moves = game_mgr.fn_get_valid_moves(board, 1)
        if valid_moves is None:
            return None
        while valid_moves[action]!=1:
            action = np.random.randint(game_mgr.fn_get_action_size())
        return action
    return fn_get_action