from collections import namedtuple

from ws.RLAgents.self_play.alpha_zero.search.non_recursive.Mcts import Mcts
from ws.RLAgents.self_play.alpha_zero.search.recursive.MCTS import MCTS


def mcts_adapter(game, nnet, args):

    if not args.mcts_recursive:
        fn_predict_action_probablities = nnet.predict
        fn_get_valid_actions = lambda board: game.fn_get_valid_moves(board, 1)

        # fn_get_action_size = game.fn_get_action_size  # ! game.fn_get_game_action_size

        fn_terminal_state_status = lambda pieces: game.fn_get_game_progress_status(pieces, 1)

        # def __next_state_mgt(fn_get_game_next_state_for_player, fn_get_current_state_for_player):
        #     def fn_find_next_state(state, action):
        #         next_state, next_player = fn_get_game_next_state_for_player(state, 1, action)
        #         if next_player is None:
        #             return None
        #         new_state = fn_get_current_state_for_player(next_state, next_player)
        #         return new_state
        #
        #     return fn_find_next_state
        #
        # fn_find_next_state = __next_state_mgt(game.fn_get_next_state, game.fn_get_canonical_form)

        mcts = Mcts(
            fn_get_next_state = game.fn_get_next_state,
            fn_get_canonical_form = game.fn_get_canonical_form,
            fn_predict_action_probablities=fn_predict_action_probablities,
            fn_get_valid_actions=fn_get_valid_actions,
            fn_terminal_state_status= fn_terminal_state_status,
            num_mcts_simulations=args.num_of_mc_simulations,
            explore_exploit_ratio=args.cpuct_exploration_exploitation_factor,
            max_num_actions=game.fn_get_action_size()
        )
    else:
        mcts = MCTS(game, nnet, args)
    fn_get_action_probabilities = lambda state, spread_probabilities: mcts.fn_get_action_probabilities(state, spread_probabilities)

    mtcs_adapter = namedtuple('_', ['fn_get_action_probabilities'])
    mtcs_adapter.fn_get_action_probabilities=fn_get_action_probabilities

    return mtcs_adapter





