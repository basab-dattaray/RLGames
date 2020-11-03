# PURPOSE: to reduce cost for taking the near optimal action (edge) when there are many possible actions
# The number of num_mcts_simulations determines how thorough a search you want.
# Remember that the higher the num_mcts_simulations, the better estimate because there will be more monte carlo rollouts for better estimation.
# A explore_exploit_ratio tells the search to put higher emphasis on exploration relative to exploitation.
# The more evolved board_pieces states (nodes) will have fewer allowable actions (edges)
from collections import namedtuple

import numpy

from .Node import Node

# from .rollout_mgt import rollout_mgt
from ..mcts_probability_mgt import mcts_probability_mgt

MTCS_RESULTS_FILE_NAME = 'mtcs_results.pkl'
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

    root_node = None

    def fn_rollout(state):
        EPS = 1e-8

        def _fn_get_state_info(fn_terminal_value, state):
            qval = None

            terminal_state = False
            if fn_terminal_value is not None:
                qval = fn_terminal_value(state)
                if qval != 0:
                    terminal_state = True
                    return -qval, None, terminal_state

            action_probabilities, state_value = fn_get_normalized_predictions(state)[:-1]

            return state_value[0], action_probabilities, terminal_state

        def _fn_get_best_action(state, action_probs):
            best_action = numpy.random.choice(len(action_probs), p=action_probs)

            next_state, next_player = fn_get_next_state(state, 1, best_action)
            next_state_canonical = fn_get_canonical_form(next_state, next_player)
            return next_state_canonical

        opponent_val, action_probs, is_terminal_state = _fn_get_state_info(
            fn_terminal_value, state
        )
        while not is_terminal_state:
            next_state = _fn_get_best_action(state, action_probs)
            opponent_val, action_probs, is_terminal_state = _fn_get_state_info(
                fn_terminal_value, next_state)
            state = next_state

        return -opponent_val, is_terminal_state

    def fn_init_mcts():
        nonlocal root_node
        root_node = None
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

    def fn_execute_monte_carlo_tree_search(state):
        nonlocal  root_node

        if root_node is None:
            root_node = Node(
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

        score, terminal_state = fn_rollout(selected_node.state)

        value = selected_node.fn_back_propagate(score)
        return value

    mcts_mgr = namedtuple('_', ['fn_get_action_probabilities'])
    mcts_mgr.fn_get_action_probabilities = fn_get_action_probabilities

    return mcts_mgr

