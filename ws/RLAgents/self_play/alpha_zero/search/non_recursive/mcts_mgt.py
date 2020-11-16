# PURPOSE: to reduce cost for taking the near optimal action (edge) when there are many possible actions
# The number of num_mcts_simulations determines how thorough a search you want.
# Remember that the higher the num_mcts_simulations, the better estimate because there will be more monte carlo rollouts for better estimation.
# A explore_exploit_ratio tells the search to put higher emphasis on exploration relative to exploitation.
# The more evolved board_pieces states (nodes) will have fewer allowable actions (edges)
from collections import namedtuple

from .mcts_cache_mgt import mcts_cache_mgt
from .node_mgt import node_mgt
from .rollout import fn_rollout
from ..mcts_probability_mgt import mcts_probability_mgt
from ..search_cache_mgt import search_cache_mgt

LONG_ROLLOUT = True
CACHE_RESULTS = False

def mcts_mgt(
        fn_get_normalized_predictions,
        fn_get_state_key,
        fn_get_next_state,
        fn_get_canonical_form,
        fn_terminal_value,
        num_mcts_simulations,
        explore_exploit_ratio,
        max_num_actions
):
    mcts_cache_mgr = mcts_cache_mgt(
            fn_get_state_key,
            fn_terminal_value
    )
    search_cache_mgr = search_cache_mgt()

    root_node = None
    def fn_init_mcts():
        nonlocal root_node
        root_node = None
        return True

    def fn_get_mcts_counts(state):
        def _fn_get_counts():
            if root_node is None:
                return None
            else:
                childrenNodes = root_node.children_nodes
                counts = [0] * max_num_actions
                for key, val in childrenNodes.items():
                    index = int(key)
                    counts[index] = val.visits
                return counts

        for i in range(num_mcts_simulations):
            fn_execute_monte_carlo_tree_search(state)
        counts = _fn_get_counts()
        return counts

    fn_get_action_probabilities = mcts_probability_mgt(fn_init_mcts, fn_get_mcts_counts)

    def fn_execute_monte_carlo_tree_search(state):
        nonlocal  root_node

        if root_node is None:
            root_node = node_mgt(
                state,
                fn_get_normalized_predictions,
                max_num_actions,
                explore_exploit_ratio,

                parent_action=-1,
                val=0.0,
                parent_node=None
            )

        selected_node = root_node.fn_select_from_available_leaf_nodes()

        if selected_node.fn_is_already_visited():
            selected_node = selected_node.fn_expand_node()
            if selected_node is None:
                return None
            pass

        score, terminal_state = fn_rollout(
            mcts_cache_mgr, fn_get_normalized_predictions, fn_get_next_state, fn_get_canonical_form, fn_terminal_value,
            state)

        value = selected_node.fn_back_propagate(score)
        return value

    mcts_mgr = namedtuple('_', ['fn_get_action_probabilities', 'fn_execute_monte_carlo_tree_search'])
    mcts_mgr.fn_get_action_probabilities = fn_get_action_probabilities
    mcts_mgr.fn_execute_monte_carlo_tree_search = fn_execute_monte_carlo_tree_search
    return mcts_mgr

