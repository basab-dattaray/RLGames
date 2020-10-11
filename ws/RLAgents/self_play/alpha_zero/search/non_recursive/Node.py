import math
import uuid
import numpy



class Node(object):
    # DEBUG_FLAG = False

    def __init__(self,
        fn_get_valid_normalized_action_probabilities,
        num_edges,
        explore_exploit_ratio,

        val=0.0,
        parent_node=None,
        parent_action=-1,
        state= None,

     ):
        self.fn_get_valid_normalized_action_probabilities = fn_get_valid_normalized_action_probabilities
        self.num_edges = num_edges
        self.explore_exploit_ratio = explore_exploit_ratio
        self.val = val
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.state = state

        self.visits = 0
        self.__states = set()
        self.opponent_factor = 1
        self.children_nodes = {}
        self.id = uuid.uuid4()

        if self.state is None:
            # parent_state = parent_node.state if parent_node is not None else None
            parent_state = numpy.copy(parent_node.state)
            current_state = self.__fn_compute_state(num_edges, parent_state, parent_action)
            self.state = current_state

    def __fn_compute_state(self, num_edges, parent_state, parent_action):
        if parent_action >= num_edges:
            return None
        state_dims = numpy.shape(parent_state)
        x, y = (int(parent_action / state_dims[0]), parent_action % state_dims[1])

        if parent_state[x][y] == 0:
            parent_state[x][y] = 1

        return parent_state

    def __fn_add_val_to_node(self, val):
        self.val += val
        self.visits += 1

    def __fn_add_children_nodes(self, normalized_valid_action_probabilities):

        children = {}
        for action_num, action_probability in enumerate(normalized_valid_action_probabilities[:-1]):
            if action_probability > 0:

                child_node = Node(
                    self.fn_get_valid_normalized_action_probabilities,
                    self.num_edges,
                    self.explore_exploit_ratio,

                    val=0.0,
                    parent_node= self,  # ??? cant be None
                    parent_action= action_num
                )
                children[str(action_num)] = child_node

        self.children_nodes = {**children}

        if len(children.values()) == 0:
            return None

        return list(children.values())[0]

    def __fn_find_best_ucb_child(self):
         # neuralnet update was based on previous player
        best_child = None
        best_ucb = 0

        normalized_valid_action_probabilities = self.fn_get_valid_normalized_action_probabilities(action_probabilities= None)

        for key, child in self.children_nodes.items():
            action_num = int(key)
            action_prob = normalized_valid_action_probabilities[action_num]
            parent_visits = self.visits
            child_visits = child.visits
            child_value = child.val * self.opponent_factor
            if child_visits == 0:
                return child

            exploit_val = child_value / child_visits
            explore_val = action_prob * math.sqrt(parent_visits) / (child_visits + 1)
            # explore_val = action_prob * math.sqrt(numpy.log(parent_visits) / child_visits)
            ucb = exploit_val + self.explore_exploit_ratio * explore_val # Upper Confidence Bound

            if best_child is None:
                best_child = child
                best_ucb = ucb
            else:
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

        return best_child

    def __fn_get_parent_node(self):
        return self.parent_node

    def fn_select_from_available_leaf_nodes(self):
        if len(self.children_nodes) == 0:  # leaf_node
            return self

        best_child = self.__fn_find_best_ucb_child()
        return best_child.fn_select_from_available_leaf_nodes()

    def fn_is_already_visited(self):
        if self.visits > 0:
            return True
        else:
            return False

    def fn_expand_node(self):
        normalized_valid_action_probabilities = self.fn_get_valid_normalized_action_probabilities(action_probabilities= None)
        if normalized_valid_action_probabilities is None:
            return None
        first_child_node = self.__fn_add_children_nodes(normalized_valid_action_probabilities)

        return first_child_node

    def fn_back_propagate(self, val):
        current_node = self
        self.__fn_add_val_to_node(val)

        parent_node = self.__fn_get_parent_node()

        while parent_node is not None:
            current_node = parent_node
            current_node.__fn_add_val_to_node(val)
            # print(current_node.val)
            parent_node = current_node.__fn_get_parent_node()

        return self.val





