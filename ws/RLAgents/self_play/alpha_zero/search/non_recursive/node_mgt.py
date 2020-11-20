import math
import uuid
from collections import namedtuple

import numpy


def node_mgt(
        fn_get_normalized_predictions,
        explore_exploit_ratio,
):
    def node(
            state,
            val,
            parent_node
    ):
        # DEBUG_FLAG = False

        visits = 0
        children_nodes = {}

        def _fn_add_children_nodes(normalized_valid_action_probabilities):
            nonlocal children_nodes

            action_probabilities = normalized_valid_action_probabilities[:-2][0]
            children_nodes = {}
            for action_num, action_probability in enumerate(action_probabilities):
                if action_probability > 0:
                    child_node = node(
                        state,
                        val=0.0,
                        parent_node= node_obj,  # ??? cant be None
                    )
                    children_nodes[str(action_num)] = child_node

            # children_nodes = children_nodes # {**children} #???

            if len(children_nodes.values()) == 0:
                return None

            return list(children_nodes.values())[0]


        def _fn_find_best_ucb_child():
                best_child = None
                best_ucb = 0

                normalized_predictions = fn_get_normalized_predictions(
                    state)  # fn_get_valid_normalized_action_probabilities()
                normalized_valid_action_probabilities = normalized_predictions[:-2][0]
                for key, child_node in children_nodes.items():
                    action_num = int(key)
                    action_prob = normalized_valid_action_probabilities[action_num]
                    parent_visits = visits
                    child_visits = child_node.fn_get_num_visits()
                    child_value = child_node.fn_get_node_val()
                    if child_visits == 0:
                        return child_node

                    exploit_val = child_value / child_visits
                    explore_val = action_prob * math.sqrt(parent_visits) / (child_visits + 1)
                    ucb = exploit_val + explore_exploit_ratio * explore_val  # Upper Confidence Bound

                    if best_child is None:
                        best_child = child_node
                        best_ucb = ucb
                    else:
                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_child = child_node

                return best_child


        def fn_select_from_available_leaf_nodes():

            if len(children_nodes) == 0:  # leaf_node
                return node_obj

            best_child = _fn_find_best_ucb_child()
            return best_child.fn_select_from_available_leaf_nodes()

        def fn_is_already_visited():
            if visits > 0:
                return True
            else:
                return False

        def fn_expand_node():
            normalized_valid_action_probabilities = fn_get_normalized_predictions(state) # fn_get_valid_normalized_action_probabilities()
            if normalized_valid_action_probabilities is None:
                return None
            first_child_node = _fn_add_children_nodes(normalized_valid_action_probabilities)

            return first_child_node

        def _fn_add_val_to_node(new_val):
            nonlocal visits, val

            val += new_val
            visits += 1
            return val

        def fn_back_propagate(current_val):
            current_node = node_obj

            while current_node is not None:
                current_val = _fn_add_val_to_node(current_val)
                current_node = current_node.parent_node

            return current_val

        def fn_get_num_visits():
            return visits

        def fn_get_children_nodes():
            return children_nodes

        def fn_get_node_val():
            return val


        node_obj = namedtuple('_', [
            'fn_get_num_visits',
            'fn_get_children_node',
            'fn_get_node_val',

            'fn_select_from_available_leaf_nodes',
            'fn_is_already_visited',
            'fn_back_propagate',
            'fn_expand_node',

            # '_fn_add_val_to_node',
            'parent_node'
        ])

        node_obj.fn_get_num_visits = fn_get_num_visits
        node_obj.fn_get_children_nodes = fn_get_children_nodes

        node_obj.fn_get_node_val = fn_get_node_val
        node_obj.parent_node = parent_node

        node_obj.fn_select_from_available_leaf_nodes = fn_select_from_available_leaf_nodes
        node_obj.fn_is_already_visited = fn_is_already_visited
        node_obj.fn_back_propagate = fn_back_propagate
        node_obj.fn_expand_node = fn_expand_node
        return node_obj

    node_mgr = namedtuple('_', ['node'])
    node_mgr.node = node
    return node_mgr





