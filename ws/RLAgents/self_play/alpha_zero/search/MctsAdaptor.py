class MctsAdapter():
    def __init__(self,
                    game,
                    nnet,
                    args
                 ):

        self.fn_predict_action_probablities = nnet.predict
        self.fn_get_valid_actions = lambda board: game.fn_get_valid_moves(board, 1)
        self.fn_find_next_state = self.__next_state_mgt(game.fn_get_next_state, game.fn_get_canonical_form)

        self.fn_get_action_size =  game.fn_get_action_size #! game.fn_get_game_action_size

        self.fn_terminal_state_status =  lambda board: game.fn_get_game_progress_status(board, 1)

        # self.fn_notify_game_result = game.fn_notify_game_result
        pass

    @staticmethod
    def __next_state_mgt(fn_get_game_next_state_for_player, fn_get_current_state_for_player):
        def fn_find_next_state(state, action):
            next_state, next_player  = fn_get_game_next_state_for_player(state, 1, action)
            if next_player is None:
                return None
            new_state = fn_get_current_state_for_player(next_state, next_player)
            return new_state

        return fn_find_next_state



