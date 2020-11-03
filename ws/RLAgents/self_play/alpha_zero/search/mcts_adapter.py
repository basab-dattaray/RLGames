from collections import namedtuple

import numpy as np

from ws.RLAgents.self_play.alpha_zero.search.non_recursive.mcts_mgt import mcts_mgt
from ws.RLAgents.self_play.alpha_zero.search.recursive.mcts_r_mgr import mcts_r_mgr


def mcts_adapter(game, neural_net_mgr, args):
    fn_predict_action_probablities = neural_net_mgr.predict
    fn_get_valid_actions = lambda board: game.fn_get_valid_moves(board, 1)
    fn_terminal_state_status = lambda pieces: game.fn_get_game_progress_status(pieces, 1)

    monte_carlo_tree_search = mcts_mgt
    if args.mcts_recursive:
        monte_carlo_tree_search = mcts_r_mgr
    def create_normalized_predictor (fn_predict_action_probablities, fn_get_valid_actions):
        def fn_get_normalized_predictions( state):
            pi, v = fn_predict_action_probablities(state)
            valid_actions = fn_get_valid_actions(state)
            pi = pi * valid_actions  # masking invalid moves
            sum_Ps_s = np.sum(pi)
            if sum_Ps_s > 0:
                pi /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                # log.error("All valid moves were masked, doing a workaround.")
                pi = pi + valid_actions
                pi /= np.sum(pi)
            return pi, v, valid_actions
        return fn_get_normalized_predictions

    fn_get_normalized_predictions = create_normalized_predictor (fn_predict_action_probablities, fn_get_valid_actions)

    mcts = monte_carlo_tree_search(
        fn_get_normalized_predictions = fn_get_normalized_predictions,
        fn_get_state_key = game.fn_get_state_key,
        fn_get_next_state = game.fn_get_next_state,
        fn_get_canonical_form = game.fn_get_canonical_form,
        fn_terminal_state_status= fn_terminal_state_status,
        num_mcts_simulations=args.num_of_mc_simulations,
        explore_exploit_ratio=args.cpuct_exploration_exploitation_factor,
        max_num_actions=game.fn_get_action_size()
    )
    fn_get_action_probabilities = lambda state, spread_probabilities: mcts.fn_get_action_probabilities(state, spread_probabilities)

    mtcs_adapter = namedtuple('_', ['fn_get_action_probabilities'])
    mtcs_adapter.fn_get_action_probabilities=fn_get_action_probabilities

    return mtcs_adapter





