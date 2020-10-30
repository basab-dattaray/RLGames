def MctsAdapter(game,
                    nnet,
                    args):

    fn_predict_action_probablities = nnet.predict
    fn_get_valid_actions = lambda board: game.fn_get_valid_moves(board, 1)

    fn_get_action_size =  game.fn_get_action_size #! game.fn_get_game_action_size

    fn_terminal_state_status =  lambda pieces: game.fn_get_game_progress_status(pieces, 1)

    def __next_state_mgt(fn_get_game_next_state_for_player, fn_get_current_state_for_player):
        def fn_find_next_state(state, action):
            next_state, next_player  = fn_get_game_next_state_for_player(state, 1, action)
            if next_player is None:
                return None
            new_state = fn_get_current_state_for_player(next_state, next_player)
            return new_state

        return fn_find_next_state

    fn_find_next_state = __next_state_mgt(game.fn_get_next_state, game.fn_get_canonical_form)

    

