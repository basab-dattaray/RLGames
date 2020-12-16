import copy
import math
from collections import namedtuple

import numpy


def node_mgt(
        args,
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
            visits = 0
    ):
        __ucb_id = None

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
            'state',
            'player',
            'children_nodes',
            'fn_display_tree',
            'fn_diagnostics',
        ])

        _id = id
        _state = state
        visits = visits
        value=0.0
        _children_nodes = {}
        _parent_node = copy.deepcopy(parent_node)
        __ucb_criteria = 'Infinity'

        # _player = player
        # _first_time = first_time

        def fn_diagnostics(ucb_criteria, ucb_id):
            nonlocal __ucb_criteria, __ucb_id
            __ucb_criteria = ucb_criteria
            __ucb_id = ucb_id
            pass

        def fn_display_tree(level= 1):
            leading_spaces = ' ' * level * 2
            str = f'{leading_spaces}{_id} ------ visits:{visits},   value:{value},     ucb:{__ucb_criteria}'

            if visits > 0:
                average_value = value/visits
                str = f'{str}   average value = {average_value} '

            if _id == __ucb_id:
                str =  str + '***'

            print(str)
            for k, v in _children_nodes.items():
                v.fn_display_tree(level + 1)


        def fn_get_children_node():
            return _children_nodes

        def fn_find_best_ucb_child(node):
            nonlocal value, visits, _state, _id
            best_child = None
            best_ucb = 0

            policy, state_val, _= fn_get_prediction_info(_state, 1)

            children_nodes = node.fn_get_children_node()
            for action_num, child_node in children_nodes.items():
                parent_visit_factor = visits
                if args.mcts_ucb_use_log_in_numerator:
                    parent_visit_factor = numpy.log(parent_visit_factor)

                child_visits = child_node.fn_get_num_visits()
                child_value = child_node.fn_get_node_val()
                if child_visits == 0:
                    return child_node

                action_prob_for_exploration = 1
                if args.mcts_ucb_use_action_prob_for_exploration:
                    action_prob_for_exploration = policy[action_num]

                exploit_val = child_value / child_visits
                explore_val = action_prob_for_exploration * math.sqrt(parent_visit_factor) / (child_visits)
                ucb = exploit_val + explore_exploit_ratio * explore_val  # Upper Confidence Bound

                if best_child is None:
                    best_child = child_node
                    best_ucb = ucb
                else:
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child_node

                child_node.fn_diagnostics((ucb, 'exploit', exploit_val, 'explore', explore_val), best_child.id)

            return best_child

        def _fn_add_children_nodes(parent):
            nonlocal value, visits, _state, _id
            parent_id = parent.id
            valid_nodes = fn_get_valid_moves(_state, player)
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
                    pass

            if len(_children_nodes) == 0:
                return None

            first_child_key= list(_children_nodes.keys())[0]
            first_child_node = _children_nodes[first_child_key]
            return first_child_node

        def fn_select_from_available_leaf_nodes():
            nonlocal first_run__mutable
            nonlocal value, visits, _state, _id
            if first_run__mutable:
                first_run__mutable = False
                # first_child_node = _fn_add_children_nodes(node_obj)

            if len(_children_nodes) == 0:  # leaf_node
                return node_obj

            best_child_node = fn_find_best_ucb_child(node_obj)
            selected_node = best_child_node.fn_select_from_available_leaf_nodes()
            return selected_node

        def fn_is_already_visited():
            if visits > 0:
                return True
            else:
                return False

        def fn_expand_node():
            first_child_node = _fn_add_children_nodes(node_obj)

            return first_child_node

        def fn_back_propagate(current_val, depth = 1):
            nonlocal value, visits, _state, _id

            value = value + current_val
            visits += 1

            parent = fn_get_parent_node()
            if parent is None:
                return depth
            else:
                return parent.fn_back_propagate(current_val, depth + 1)

        def fn_get_num_visits():
            return visits

        def fn_get_children_nodes():
            return _children_nodes

        def fn_get_node_val():
            return value

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
        node_obj.player = player
        node_obj.children_nodes = _children_nodes
        node_obj.fn_display_tree = fn_display_tree
        node_obj.fn_diagnostics = fn_diagnostics
        return node_obj

    node_mgr = namedtuple('_', ['node'])
    node_mgr.node = node
    return node_mgr





