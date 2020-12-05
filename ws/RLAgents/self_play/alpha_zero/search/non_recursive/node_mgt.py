import math
import uuid
from collections import namedtuple

import numpy


def node_mgt(
        fn_find_best_ucb_child,
        explore_exploit_ratio,
        max_num_actions,
):
    def node(
            state,
            val = 0.0,
            parent_node = None
    ):
        visits = 0
        children_nodes = {}

        def _fn_add_children_nodes():
            nonlocal children_nodes

            # policy = normalized_valid_policy[:-2][0]
            children_nodes = {}
            for action_num in range(max_num_actions):
                child_node = node(
                    state,
                    val=0.0,
                    parent_node= node_obj,  # ??? cant be None
                )
                children_nodes[str(action_num)] = child_node

            if len(children_nodes.values()) == 0:
                return None

            return list(children_nodes.values())[0]


        def fn_select_from_available_leaf_nodes():

            if len(children_nodes) == 0:  # leaf_node
                return node_obj

            best_child = fn_find_best_ucb_child(state, children_nodes, visits, explore_exploit_ratio)
            return best_child.fn_select_from_available_leaf_nodes()

        def fn_is_already_visited():
            if visits > 0:
                return True
            else:
                return False

        def fn_expand_node():
            first_child_node = _fn_add_children_nodes()

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
                new_node = current_node.fn_get_parent_node()
                current_node = new_node

            return current_val

        def fn_get_num_visits():
            return visits

        def fn_get_children_nodes():
            return children_nodes

        def fn_get_node_val():
            return val

        def fn_get_parent_node():
            return parent_node

        node_obj = namedtuple('_', [
            'fn_get_num_visits',
            'fn_get_children_node',
            'fn_get_node_val',

            'fn_select_from_available_leaf_nodes',
            'fn_is_already_visited',
            'fn_back_propagate',
            'fn_expand_node',
            'fn_get_parent_node'
        ])

        node_obj.fn_get_num_visits = fn_get_num_visits
        node_obj.fn_get_children_nodes = fn_get_children_nodes
        node_obj.fn_get_node_val = fn_get_node_val
        node_obj.fn_get_parent_node = fn_get_parent_node
        node_obj.fn_select_from_available_leaf_nodes = fn_select_from_available_leaf_nodes
        node_obj.fn_is_already_visited = fn_is_already_visited
        node_obj.fn_back_propagate = fn_back_propagate
        node_obj.fn_expand_node = fn_expand_node

        return node_obj

    node_mgr = namedtuple('_', ['node'])
    node_mgr.node = node
    return node_mgr





