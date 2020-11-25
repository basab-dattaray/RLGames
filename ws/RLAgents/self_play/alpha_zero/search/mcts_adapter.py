from collections import namedtuple

import numpy as np

from ws.RLAgents.self_play.alpha_zero.search.non_recursive.mcts_mgt import mcts_mgt
from ws.RLAgents.self_play.alpha_zero.search.recursive.mcts_r_mgr import mcts_r_mgr


def mcts_adapter(neural_net_mgr, args):
    game_mgr = args.game_mgr
    fn_predict_action_probablities = neural_net_mgr.predict
    fn_get_valid_actions = lambda board: game_mgr.fn_get_valid_moves(board, 1)
    fn_terminal_value = lambda pieces: game_mgr.fn_get_game_progress_status(pieces, 1)

    monte_carlo_tree_search = mcts_mgt
    if args.run_recursive_search:
        monte_carlo_tree_search = mcts_r_mgr
    def create_normalized_predictor (fn_predict_action_probablities, fn_get_valid_actions):
        def fn_get_prediction_info( state):
            action_probalities, wrapped_state_val = fn_predict_action_probablities(state)
            valid_actions = fn_get_valid_actions(state)
            action_probalities = action_probalities * valid_actions  # masking invalid moves
            sum_Ps_s = np.sum(action_probalities)
            if sum_Ps_s > 0:
                action_probalities /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.

                action_probalities = action_probalities + valid_actions
                action_probalities /= np.sum(action_probalities)
            return action_probalities, wrapped_state_val[0], valid_actions
        return fn_get_prediction_info

    fn_get_prediction_info = create_normalized_predictor (fn_predict_action_probablities, fn_get_valid_actions)

    mcts = monte_carlo_tree_search(
        fn_get_prediction_info = fn_get_prediction_info,
        fn_get_state_key = game_mgr.fn_get_state_key,
        fn_get_next_state = game_mgr.fn_get_next_state,
        fn_get_canonical_form = game_mgr.fn_get_canonical_form,
        fn_terminal_value= fn_terminal_value,
        num_mcts_simulations=args.num_of_mc_simulations,
        explore_exploit_ratio=args.cpuct_exploration_exploitation_factor,
        max_num_actions=game_mgr.fn_get_action_size()
    )
    fn_get_policy = lambda state, spread_probabilities: mcts.fn_get_policy(state, spread_probabilities)

    mtcs_adapter = namedtuple('_', ['fn_get_policy', 'fn_get_prediction_info', 'fn_terminal_value'])
    mtcs_adapter.fn_get_policy=fn_get_policy
    mtcs_adapter.fn_get_prediction_info=fn_get_prediction_info
    mtcs_adapter.fn_terminal_value = fn_terminal_value
    return mtcs_adapter





