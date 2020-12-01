from collections import namedtuple

# import numpy as np

from ws.RLAgents.self_play.alpha_zero.search.non_recursive.mcts_mgt import mcts_mgt
from ws.RLAgents.self_play.alpha_zero.search.recursive.mcts_r_mgr import mcts_r_mgr

def mcts_adapter(neural_net_mgr, args):
    game_mgr = args.game_mgr

    monte_carlo_tree_search = mcts_mgt
    if args.run_recursive_search:
        monte_carlo_tree_search = mcts_r_mgr

    mcts = monte_carlo_tree_search(
        game_mgr,
        neural_net_mgr,
        num_mcts_simulations=args.num_of_mc_simulations,
        explore_exploit_ratio=args.cpuct_exploration_exploitation_factor,
        max_num_actions=game_mgr.fn_get_action_size()
    )
    fn_get_policy = lambda state, spread_probabilities: mcts.fn_get_policy(state, spread_probabilities)

    mtcs_adapter = namedtuple('_', ['fn_get_policy', 'fn_get_prediction_info', 'fn_terminal_value'])
    mtcs_adapter.fn_get_policy=fn_get_policy
    return mtcs_adapter





