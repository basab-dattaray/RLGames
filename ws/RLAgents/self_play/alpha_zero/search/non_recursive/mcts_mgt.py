# PURPOSE: to reduce cost for taking the near optimal action (edge) when there are many possible actions
# The number of num_mcts_simulations determines how thorough a search you want.
# Remember that the higher the num_mcts_simulations, the better estimate because there will be more monte carlo rollouts for better estimation.
# A explore_exploit_ratio tells the search to put higher emphasis on exploration relative to exploitation.
# The more evolved board_pieces states (nodes) will have fewer allowable actions (edges)
from collections import namedtuple

import numpy

from .action_mgt import action_mgt
from .node_mgt import node_mgt
from ..cache_mgt import cache_mgt
from ..policy_mgt import policy_mgt
from ws.RLAgents.self_play.alpha_zero.search.non_recursive.cache2_mgt import cache2_mgt

USE_SMART_PREDICTOR_FOR_ROLLOUT = False

def mcts_mgt(
        args,
        game_mgr,
        neural_net_mgr,
        playground_mgt,
        num_mcts_simulations,
        explore_exploit_ratio,
        max_num_actions
):
    cache_mgr = cache_mgt()
    fn_get_prediction_info, fn_get_valid_moves = cache2_mgt(game_mgr, cache_mgr, neural_net_mgr)

    node_mgr = node_mgt(
        args,
        fn_get_valid_moves,
        fn_get_prediction_info,
        game_mgr.fn_get_next_state,
        explore_exploit_ratio,
        first_run__mutable= True
    )

    fn_get_action_given_state = action_mgt(USE_SMART_PREDICTOR_FOR_ROLLOUT, fn_get_valid_moves, fn_get_prediction_info)

    playground = playground_mgt(
        fn_get_action_given_state,
        fn_get_action_given_state,
        game_mgr
    )
    root_node = None

    def fn_get_mcts_counts(state):

        def _fn_get_counts():

            if root_node is None:
                return None
            else:
                children_nodes = root_node.fn_get_children_nodes()
                counts = [0] * max_num_actions
                for index, current_node in children_nodes.items():
                    visits = current_node.fn_get_num_visits()
                    counts[index] = visits
                return counts

        for i in range(num_mcts_simulations):
            fn_execute_search(state)
        counts = _fn_get_counts()
        sum_counts = sum(counts)
        return counts

    fn_get_policy = policy_mgt(fn_get_mcts_counts)

    def fn_execute_search(state):
        nonlocal  root_node

        def fn_rollout(node):
            result = playground.fn_play_one_game(node.state, turn= node.player, verbose=False)
            return result

        if root_node is None:
            root_node = node_mgr.node(
                state,
                parent_node = None,
                player = 1,
                id = '0',
                visits= 1,
            )

        selected_node = root_node.fn_select_from_available_leaf_nodes()

        if selected_node.fn_is_already_visited():
            selected_node = selected_node.fn_expand_node()
            if selected_node is None:
                return None
            pass

        score = fn_rollout(selected_node)
        tree_depth = selected_node.fn_back_propagate(score)
        pass

    mcts_mgr = namedtuple('_', ['fn_get_policy'])
    mcts_mgr.fn_get_policy = fn_get_policy
    return mcts_mgr



