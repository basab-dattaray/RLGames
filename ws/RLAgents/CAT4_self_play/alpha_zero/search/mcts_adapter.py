from collections import namedtuple
from ws.RLAgents.CAT4_self_play.alpha_zero.search.monte_carlo_tree_search_mgt import monte_carlo_tree_search_mgt


def mcts_adapter(neural_net_mgr, args):
    game_mgr = args.game_mgr

    mcts = monte_carlo_tree_search_mgt(
        game_mgr,
        neural_net_mgr,
        args,
    )
    fn_get_policy = lambda state, do_random_selection: mcts.fn_get_policy(state, do_random_selection)

    mtcs_adapter = namedtuple('_', ['fn_get_policy'])
    mtcs_adapter.fn_get_policy=fn_get_policy
    return mtcs_adapter





