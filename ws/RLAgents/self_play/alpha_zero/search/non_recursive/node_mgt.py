import copy
import math
from collections import namedtuple

def node_mgt(
        fn_get_valid_moves,
        fn_get_prediction_info,
        fn_get_next_state,
        explore_exploit_ratio,
        first_run__mutable
):
    def node(
            state,
            parent_node,
            player,
            id,
    ):
        node_obj = namedtuple('_', [
            'fn_get_num_visits',
            'fn_get_children_node',
            'fn_get_node_val',

            'fn_select_from_available_leaf_nodes',
            'fn_is_already_visited',
            'fn_back_propagate',
            'fn_expand_node',
            'fn_get_parent_node',
            'id',
            'state'

        ])

        _id = id
        _state = state
        _visits = 0
        _val=0.0
        _children_nodes = {}
        _parent_node = copy.deepcopy(parent_node)
        # _first_time = first_time

        def fn_get_children_node():
            return _children_nodes

        def fn_find_best_ucb_child(node):
            nonlocal _val, _visits, _state, _id
            best_child = None
            best_ucb = 0

            policy, state_val, _= fn_get_prediction_info(_state, 1)

            children_nodes = node.fn_get_children_node()
            for action_num, child_node in children_nodes.items():

                action_prob = policy[action_num]

                child_visits = child_node.fn_get_num_visits()
                child_value = child_node.fn_get_node_val()
                if child_visits == 0:
                    return child_node

                exploit_val = child_value / child_visits
                explore_val = action_prob * math.sqrt(_visits) / (child_visits)
                ucb = exploit_val + explore_exploit_ratio * explore_val  # Upper Confidence Bound

                if best_child is None:
                    best_child = child_node
                    best_ucb = ucb
                else:
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child_node

            return best_child

        def _fn_add_children_nodes(parent):
            nonlocal _val, _visits, _state, _id
            parent_id = parent.id
            valid_nodes = fn_get_valid_moves(_state, 1)
            if valid_nodes is None:
                return None
            for action, valid in enumerate(valid_nodes):
                if valid != 0:
                    new_id = parent_id + '.' + str(action)
                    new_state = fn_get_next_state(_state, player, action)
                    child_node = node(
                        state = new_state,
                        parent_node = parent,
                        player = player * -1,
                        id = new_id
                    )
                    _children_nodes[action] =  child_node

            if len(_children_nodes) == 0:
                return None

            first_child_key= list(_children_nodes.keys())[0]
            first_child_node = _children_nodes[first_child_key]
            return first_child_node

        def fn_select_from_available_leaf_nodes():
            nonlocal first_run__mutable
            nonlocal _val, _visits, _state, _id
            if first_run__mutable:
                first_run__mutable = False
                first_child_node = _fn_add_children_nodes(node_obj)

            if len(_children_nodes) == 0:  # leaf_node
                return node_obj

            best_child_node = fn_find_best_ucb_child(node_obj)
            selected_node = best_child_node.fn_select_from_available_leaf_nodes()
            return selected_node

        def fn_is_already_visited():
            if _visits > 0:
                return True
            else:
                return False

        def fn_expand_node():
            first_child_node = _fn_add_children_nodes(node_obj)

            return first_child_node

        def fn_back_propagate(current_val, depth = 1):
            nonlocal _val, _visits, _state, _id

            _val = _val + current_val
            _visits += 1

            parent = fn_get_parent_node()
            if parent is None:
                return depth
            else:
                return parent.fn_back_propagate(current_val, depth + 1)

        def fn_get_num_visits():
            return _visits

        def fn_get_children_nodes():
            return _children_nodes

        def fn_get_node_val():
            return _val

        def fn_get_parent_node():
            return parent_node


        node_obj.fn_get_children_node = fn_get_children_node
        node_obj.fn_get_num_visits = fn_get_num_visits
        node_obj.fn_get_children_nodes = fn_get_children_nodes
        node_obj.fn_get_node_val = fn_get_node_val
        node_obj.fn_get_parent_node = fn_get_parent_node
        node_obj.fn_select_from_available_leaf_nodes = fn_select_from_available_leaf_nodes
        node_obj.fn_is_already_visited = fn_is_already_visited
        node_obj.fn_back_propagate = fn_back_propagate
        node_obj.fn_expand_node = fn_expand_node
        node_obj.id = id
        node_obj.state = state

        return node_obj

    node_mgr = namedtuple('_', ['node'])
    node_mgr.node = node
    return node_mgr





