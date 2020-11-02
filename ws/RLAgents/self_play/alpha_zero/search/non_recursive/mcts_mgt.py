# PURPOSE: to reduce cost for taking the near optimal action (edge) when there are many possible actions
# The number of num_mcts_simulations determines how thorough a search you want.
# Remember that the higher the num_mcts_simulations, the better estimate because there will be more monte carlo rollouts for better estimation.
# A explore_exploit_ratio tells the search to put higher emphasis on exploration relative to exploitation.
# The more evolved board_pieces states (nodes) will have fewer allowable actions (edges)
from collections import namedtuple

from .Node import Node

from .rollout_mgt import rollout_mgt
from ..mcts_probability_mgt import mcts_probability_mgt

MTCS_RESULTS_FILE_NAME = 'mtcs_results.pkl'
CACHE_RESULTS = False

def mcts_mgt(
        fn_get_normalized_predictions,
        fn_get_state_key,
        fn_get_next_state,
        fn_get_canonical_form,
        fn_predict_action_probablities,
        fn_get_valid_actions,
        fn_terminal_state_status,
        num_mcts_simulations,
        explore_exploit_ratio,
        max_num_actions
):

    # app_path =  os.getcwd()
    root_node = None
    # state_cache = None

    def fn_init_mcts(state):
        nonlocal root_node #, state_cache
        root_node = None
        # state_cache = state_cache_mgt(fn_get_valid_actions, fn_predict_action_probablities, state)
        return True

    def fn_get_mcts_counts(state):
        nonlocal root_node
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
    fn_rollout = rollout_mgt(fn_predict_action_probablities, fn_terminal_state_status,
                fn_get_next_state, fn_get_canonical_form,
                multirun=False)

    def fn_execute_monte_carlo_tree_search(state):
        nonlocal  root_node
        # if state_cache is None:
        #     return None
        # else:
        if root_node is None:
            root_node = Node(
                state,
                fn_get_normalized_predictions,
                # state_cache.fn_get_valid_normalized_action_probabilities,
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

        score, terminal_state = fn_rollout(selected_node.state)

        value = selected_node.fn_back_propagate(score)
        return value

    mcts_mgr = namedtuple('_', ['fn_get_action_probabilities'])
    mcts_mgr.fn_get_action_probabilities = fn_get_action_probabilities

    return mcts_mgr

