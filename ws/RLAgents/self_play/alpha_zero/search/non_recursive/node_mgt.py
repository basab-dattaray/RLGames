import math
import uuid
from collections import namedtuple

import numpy




def Node(
    fn_get_valid_normalized_action_probabilities,
    num_edges,
    explore_exploit_ratio,

    val=0.0,
    parent_node=None,
    parent_action=-1,
    state= None
):

    visits = 0
    __states = set()
    opponent_factor = 1
    children_nodes = {}
    id = uuid.uuid4()

    def __fn_compute_state(num_edges, parent_state, parent_action):
        if parent_action >= num_edges:
            return None
        state_dims = numpy.shape(parent_state)
        x, y = (int(parent_action / state_dims[0]), parent_action % state_dims[1])

        if parent_state[x][y] == 0:
            parent_state[x][y] = 1

        return parent_state

    def __fn_add_val_to_node(val):
        nonlocal visits
        val += val
        visits += 1

    def __fn_add_children_nodes(normalized_valid_action_probabilities):

        children = {}
        for action_num, action_probability in enumerate(normalized_valid_action_probabilities[:-1]):
            if action_probability > 0:

                child_node = Node(
                    fn_get_valid_normalized_action_probabilities,
                    num_edges,
                    explore_exploit_ratio,

                    val=0.0,
                    parent_node= ref_obj,  # ??? cant be None
                    parent_action= action_num
                )
                children[str(action_num)] = child_node

        # children_nodes = children # {**children} #???

        if len(children.values()) == 0:
            return None

        return list(children.values())[0]

    def __fn_find_best_ucb_child():
         # neuralnet update was based on previous player
        best_child = None
        best_ucb = 0

        normalized_valid_action_probabilities = fn_get_valid_normalized_action_probabilities(action_probabilities= None)

        for key, child in children_nodes.items():
            action_num = int(key)
            action_prob = normalized_valid_action_probabilities[action_num]
            parent_visits = visits
            child_visits = child.visits
            child_value = child.val * opponent_factor
            if child_visits == 0:
                return child

            exploit_val = child_value / child_visits
            explore_val = action_prob * math.sqrt(parent_visits) / (child_visits + 1)
            # explore_val = action_prob * math.sqrt(numpy.log(parent_visits) / child_visits)
            ucb = exploit_val + explore_exploit_ratio * explore_val # Upper Confidence Bound

            if best_child is None:
                best_child = child
                best_ucb = ucb
            else:
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

        return best_child

    def __fn_get_parent_node():
        return parent_node

    def fn_select_from_available_leaf_nodes():
        if len(children_nodes) == 0:  # leaf_node
            return ref_obj

        best_child = __fn_find_best_ucb_child()
        return best_child.fn_select_from_available_leaf_nodes()

    def fn_is_already_visited():
        if visits > 0:
            return True
        else:
            return False

    def fn_expand_node():
        normalized_valid_action_probabilities = fn_get_valid_normalized_action_probabilities(action_probabilities= None)
        if normalized_valid_action_probabilities is None:
            return None
        first_child_node = __fn_add_children_nodes(normalized_valid_action_probabilities)

        return first_child_node

    def fn_back_propagate(val):

        __fn_add_val_to_node(val)

        # parent_node = self.__fn_get_parent_node()
        current_node = parent_node

        while current_node is not None:
            current_node.__fn_add_val_to_node(val)
            current_node = current_node.parent_node

        return val

    if state is None:
        state = __fn_compute_state(num_edges, parent_node.state, parent_action)

    ref_obj = namedtuple('x', ['fn_select_from_available_leaf_nodes', 'fn_is_already_visited', 'fn_expand_node', 'fn_back_propagate'])
    ref_obj.fn_select_from_available_leaf_nodes=fn_select_from_available_leaf_nodes
    ref_obj.fn_is_already_visited = fn_is_already_visited
    ref_obj.fn_expand_node = fn_expand_node
    ref_obj.fn_back_propagate = fn_back_propagate

    return ref_obj

