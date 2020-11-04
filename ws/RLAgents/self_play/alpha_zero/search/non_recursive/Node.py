import math
import uuid
import numpy



class Node(object):
    # DEBUG_FLAG = False

    def __init__(self,
        state,
        fn_get_normalized_predictions,
        num_edges,
        explore_exploit_ratio,
        parent_action=-1,
        val=0.0,
        parent_node=None
     ):
        self.fn_get_normalized_predictions = fn_get_normalized_predictions
        self.num_edges = num_edges
        self.explore_exploit_ratio = explore_exploit_ratio
        self.val = val
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.state = state

        self.visits = 0
        self.children_nodes = {}
        self.id = uuid.uuid4()

    def _fn_add_val_to_node(self, val):
        self.val += val
        self.visits += 1
        return self.val

    def _fn_add_children_nodes(self, normalized_valid_action_probabilities):
        action_probabilities = normalized_valid_action_probabilities[:-2][0]
        children = {}
        for action_num, action_probability in enumerate(action_probabilities):
            if action_probability > 0:
                child_node = Node(
                    self.state,
                    self.fn_get_normalized_predictions,
                    self.num_edges,
                    self.explore_exploit_ratio,

                    val=0.0,
                    parent_node= self,  # ??? cant be None
                    parent_action= action_num
                )
                children[str(action_num)] = child_node

        self.children_nodes = children # {**children} #???

        if len(children.values()) == 0:
            return None

        return list(children.values())[0]

    def _fn_find_best_ucb_child(self):
        best_child = None
        best_ucb = 0

        normalized_predictions = self.fn_get_normalized_predictions(self.state) # self.fn_get_valid_normalized_action_probabilities()
        normalized_valid_action_probabilities = normalized_predictions[:-2][0]
        for key, child in self.children_nodes.items():
            action_num = int(key)
            action_prob = normalized_valid_action_probabilities[action_num]
            parent_visits = self.visits
            child_visits = child.visits
            child_value = child.val
            if child_visits == 0:
                return child

            exploit_val = child_value / child_visits
            explore_val = action_prob * math.sqrt(parent_visits) / (child_visits + 1)
            ucb = exploit_val + self.explore_exploit_ratio * explore_val # Upper Confidence Bound

            if best_child is None:
                best_child = child
                best_ucb = ucb
            else:
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

        return best_child

    def fn_select_from_available_leaf_nodes(self):
        if len(self.children_nodes) == 0:  # leaf_node
            return self

        best_child = self._fn_find_best_ucb_child()
        return best_child.fn_select_from_available_leaf_nodes()

    def fn_is_already_visited(self):
        if self.visits > 0:
            return True
        else:
            return False

    def fn_expand_node(self):
        normalized_valid_action_probabilities = self.fn_get_normalized_predictions(self.state) # self.fn_get_valid_normalized_action_probabilities()
        if normalized_valid_action_probabilities is None:
            return None
        first_child_node = self._fn_add_children_nodes(normalized_valid_action_probabilities)

        return first_child_node

    def fn_back_propagate(self, val):
        current_node = self
        val = self._fn_add_val_to_node(val)

        parent_node = self.parent_node

        while parent_node is not None:
            current_node = parent_node
            val = current_node._fn_add_val_to_node(val)
            parent_node = current_node.parent_node

        return val





