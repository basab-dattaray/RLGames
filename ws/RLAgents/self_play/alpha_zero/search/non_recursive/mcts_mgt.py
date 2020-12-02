# PURPOSE: to reduce cost for taking the near optimal action (edge) when there are many possible actions
# The number of num_mcts_simulations determines how thorough a search you want.
# Remember that the higher the num_mcts_simulations, the better estimate because there will be more monte carlo rollouts for better estimation.
# A explore_exploit_ratio tells the search to put higher emphasis on exploration relative to exploitation.
# The more evolved board_pieces states (nodes) will have fewer allowable actions (edges)
from collections import namedtuple

import numpy
from utils.lists import flatten

from .node_mgt import node_mgt
from ..cache_mgt import cache_mgt
from ..policy_mgt import policy_mgt
from ..prediction_mgt import prediction_mgt
from ..search_helper import create_normalized_predictor

LONG_ROLLOUT = True
CACHE_RESULTS = False
EPS = 1e-8

def mcts_mgt(
        game_mgr,
        neural_net_mgr,
        num_mcts_simulations,
        explore_exploit_ratio,
        max_num_actions
):
    # fn_terminal_value = lambda pieces: game_mgr.fn_get_game_progress_status(pieces, 1)
    # fn_get_valid_actions = lambda board: game_mgr.fn_get_valid_moves(board, 1)
    # fn_get_prediction_info = create_normalized_predictor (neural_net_mgr.predict, fn_get_valid_actions)
    cache_mgr = cache_mgt()
    fn_get_prediction_info = prediction_mgt(game_mgr, cache_mgr, neural_net_mgr)

    node_mgr = node_mgt(
        fn_get_prediction_info,
        explore_exploit_ratio
    )

    root_node = None

    def fn_get_mcts_counts(state):

        def _fn_get_counts():

            if root_node is None:
                return None
            else:
                childrenNodes = root_node.fn_get_children_nodes()
                counts = [0] * max_num_actions
                for key, current_node in childrenNodes.items():
                    index = int(key)
                    counts[index] = current_node.fn_get_num_visits()
                return counts

        for i in range(num_mcts_simulations):
            fn_execute_search(state)
        counts = _fn_get_counts()
        return counts

    fn_get_policy = policy_mgt(fn_get_mcts_counts)

    def fn_execute_search(state):
        nonlocal  root_node

        def fn_rollout(state_this):
            def _fn_get_state_stats(state):
                zeros_in_state = len(list(filter(lambda e: e == 0, numpy.array(state).flatten())))
                minuses_in_state = len(list(filter(lambda e: e == -1, numpy.array(state).flatten())))
                plusses_in_state = len(list(filter(lambda e: e == 1, numpy.array(state).flatten())))
                return zeros_in_state, minuses_in_state, plusses_in_state

            player_this = 1
            action_probs, state_result, _ = fn_get_prediction_info(state_this, player_this)

            i_ = 0

            while action_probs is not None:
                best_action = numpy.random.choice(len(action_probs), p=action_probs)

                state_next, player_next = game_mgr.fn_get_next_state(state_this, player_this, best_action)

                this_state_stats = _fn_get_state_stats(state_this)
                next_state_stats = _fn_get_state_stats(state_next)

                if this_state_stats == next_state_stats:
                    break


                state_this = state_next
                player_this = player_next

                state_next_canonical = game_mgr.fn_get_canonical_form(state_this, player_this)
                action_probs, state_result, _ = fn_get_prediction_info(state_next_canonical, player_this) # caonical state is needed for net prediction

                i_ += 1

            _, minusses, plusses = _fn_get_state_stats(state_this)
            raw_result = 0 if plusses == minusses else 1 if plusses > minusses else -1
            player_based_result = raw_result * player_this

            return player_based_result

        if root_node is None:
            root_node = node_mgr.node(
                state
            )

        selected_node = root_node.fn_select_from_available_leaf_nodes()

        if selected_node.fn_is_already_visited():
            selected_node = selected_node.fn_expand_node()
            if selected_node is None:
                return None
            pass

        score = fn_rollout(state)

        value = selected_node.fn_back_propagate(score)
        return value

    mcts_mgr = namedtuple('_', ['fn_get_policy'])
    mcts_mgr.fn_get_policy = fn_get_policy
    return mcts_mgr

