from collections import namedtuple

from ws.RLAgents.CAT4_self_play.alpha_zero.play.playground_mgt import playground_mgt
from ws.RLAgents.CAT4_self_play.alpha_zero.search.recursive.mcts_r_mgr import mcts_r_mgr


def mcts_adapter(neural_net_mgr, args):
    game_mgr = args.game_mgr


    monte_carlo_tree_search = mcts_r_mgr

    mcts = monte_carlo_tree_search(
        args,
        game_mgr,
        neural_net_mgr,
        playground_mgt,
        num_mcts_simulations=args.num_of_mc_simulations,
        cpuct_exploration_exploitation_factor=args.cpuct_exploration_exploitation_factor,
        max_num_actions=game_mgr.fn_get_action_size()
    )
    fn_get_policy = lambda state, do_random_selection: mcts.fn_get_policy(state, do_random_selection)

    mtcs_adapter = namedtuple('_', ['fn_get_policy'])
    mtcs_adapter.fn_get_policy=fn_get_policy
    return mtcs_adapter





