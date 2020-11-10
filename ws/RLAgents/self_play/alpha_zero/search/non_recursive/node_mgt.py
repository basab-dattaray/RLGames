import math
import uuid
from collections import namedtuple

import numpy



def node_mgt(
        state,
        fn_get_normalized_predictions,
        num_edges,
        explore_exploit_ratio,
        parent_action,
        val,
        parent_node
):
    # DEBUG_FLAG = False

    visits = 0
    children_nodes = {}
    id = uuid.uuid4()

    def fn_add_val_to_node(new_val):
        nonlocal  visits, val
        
        val += new_val
        visits += 1
        return val

    def _fn_add_children_nodes(normalized_valid_action_probabilities):
        nonlocal parent_action

        action_probabilities = normalized_valid_action_probabilities[:-2][0]
        children = {}
        for action_num, action_probability in enumerate(action_probabilities):
            if action_probability > 0:
                child_node = node_mgt(
                    state,
                    fn_get_normalized_predictions,
                    num_edges,
                    explore_exploit_ratio,

                    val=0.0,
                    parent_node= node_mgr,  # ??? cant be None
                    parent_action= action_num
                )
                children[str(action_num)] = child_node

        children_nodes = children # {**children} #???

        if len(children.values()) == 0:
            return None

        return list(children.values())[0]

    def _fn_find_best_ucb_child():
        best_child = None
        best_ucb = 0

        normalized_predictions = fn_get_normalized_predictions(state) # fn_get_valid_normalized_action_probabilities()
        normalized_valid_action_probabilities = normalized_predictions[:-2][0]
        for key, child in children_nodes.items():
            action_num = int(key)
            action_prob = normalized_valid_action_probabilities[action_num]
            parent_visits = visits
            child_visits = child.visits
            child_value = child.val
            if child_visits == 0:
                return child

            exploit_val = child_value / child_visits
            explore_val = action_prob * math.sqrt(parent_visits) / (child_visits + 1)
            ucb = exploit_val + explore_exploit_ratio * explore_val # Upper Confidence Bound

            if best_child is None:
                best_child = child
                best_ucb = ucb
            else:
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

        return best_child

    def fn_select_from_available_leaf_nodes():
        if len(children_nodes) == 0:  # leaf_node
            return node_mgr

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

    # def fn_back_propagate(current_val):
    #     nonlocal parent_node
    #
    #     current_node = node_mgr
    #     current_val = fn_add_val_to_node(current_val)
    #
    #     # parent_node = parent_node
    #
    #     while parent_node is not None:
    #         current_node = parent_node
    #         current_val = current_node.fn_add_val_to_node(current_val)
    #         parent_node = current_node.parent_node
    #
    #     return current_val

    def fn_back_propagate(current_val):

        node = node_mgr

        while node is not None:
            current_val = node.fn_add_val_to_node(current_val)
            node = node.parent_node

        return current_val

    
    node_mgr = namedtuple('_', [
        'visits',
        'children_nodes',
        'id',
        'state',
        'current_val',

        'fn_select_from_available_leaf_nodes',
        'fn_is_already_visited',
        'fn_back_propagate',
        'fn_expand_node',

        'fn_add_val_to_node',
        'parent_node',

    ])

    node_mgr.visits = visits
    node_mgr.children_nodes = children_nodes
    node_mgr.id = id
    node_mgr.state = state
    node_mgr.val = val
    node_mgr.parent_node = parent_node

    node_mgr.fn_select_from_available_leaf_nodes = fn_select_from_available_leaf_nodes
    node_mgr.fn_is_already_visited = fn_is_already_visited
    node_mgr.fn_back_propagate = fn_back_propagate
    node_mgr.fn_expand_node = fn_expand_node

    node_mgr.fn_add_val_to_node = fn_add_val_to_node

    return node_mgr





