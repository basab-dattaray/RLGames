import math
import uuid
import numpy

class Node(object):
    DEBUG_FLAG = False

    def __init__(self,
        ref_mcts,
        num_edges,
        explore_exploit_ratio,

        val=0.0,
        parent_node=None,
        parent_action=-1,
        state= None,

     ):
        self.__states = set()
        self.opponent_factor = 1
        self.ref_mcts = ref_mcts
        self.num_edges = num_edges
        self.explore_exploit_ratio = explore_exploit_ratio

        self.parent = parent_node
        self.visits = 0
        self.val = val
        self.children_nodes = {}
        self.id = uuid.uuid4()
        self.parent_action = parent_action

        if state is not None:
            self.state = state
        else:
            # parent_state = parent_node.state if parent_node is not None else None
            parent_state = numpy.copy(parent_node.state)
            current_state = self.__fn_compute_state(num_edges, parent_state, parent_action)
            self.state = current_state

    @staticmethod
    def __fn_compute_state(num_edges, parent_state, parent_action):
        if parent_action >= num_edges:
            return None
        state_dims = numpy.shape(parent_state)
        x, y = (int(parent_action / state_dims[0]), parent_action % state_dims[1])

        if parent_state[x][y] == 0:
            parent_state[x][y] = 1

        return parent_state

    def __fn_inspect_node_info(self):
        return self.id, self.parent, self.visits, self.val, self.children_nodes, self.parent_action

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
        normalized_valid_action_probabilities = self.ref_mcts.state_cache.fn_get_valid_normalized_action_probabilities(action_probabilities= None)
        if normalized_valid_action_probabilities is None:
            return None
        first_child_node = self.__fn_add_children_nodes(normalized_valid_action_probabilities)

        return first_child_node

    def __fn_add_val_to_node(self, val):

        self.val += val
        self.visits += 1

    def __fn_get_parent_node(self):
        return self.parent

    # def fn_rollout(self, multirun= False):
    #     state = self.state
    #
    #     rollout_impl = Rollout(self.ref_mcts.fn_predict_action_probablities)
    #
    #     opponent_val, action_probs, is_terminal_state =  rollout_impl.fn_get_rollout_value(
    #         self.ref_mcts.fn_terminal_state_status, state
    #     )
    #
    #     while not is_terminal_state and multirun:
    #         normalized_valid_action_probabilities = self.ref_mcts.state_cache.fn_get_valid_normalized_action_probabilities(
    #             action_probabilities = action_probs
    #         )
    #         if normalized_valid_action_probabilities is None:
    #             is_terminal_state = True
    #         else:
    #             action = numpy.random.choice(len(normalized_valid_action_probabilities), p=normalized_valid_action_probabilities)
    #             new_state = self.ref_mcts.fn_find_next_state(state, action)
    #             if new_state is None:
    #                 is_terminal_state = True
    #             else:
    #                 opponent_val, action_probs, is_terminal_state = rollout_impl.fn_get_rollout_value(
    #                     self.ref_mcts.fn_terminal_state_status, new_state
    #                 )
    #
    #                 state = new_state
    #     val =  -opponent_val
    #     return val, is_terminal_state

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

    def __fn_add_children_nodes(self, normalized_valid_action_probabilities):

        children = {}
        for action_num, action_probability in enumerate(normalized_valid_action_probabilities[:-1]):
            if action_probability > 0:

                child_node = Node(
                    self.ref_mcts,
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

        normalized_valid_action_probabilities = self.ref_mcts.state_cache.fn_get_valid_normalized_action_probabilities(action_probabilities= None)

        for key, child in self.children_nodes.items():
            action_num = int(key)
            action_prob = normalized_valid_action_probabilities[action_num]
            parent_visits = self.__fn_get_visits()
            child_visits = child.__fn_get_visits()
            child_value = child.__fn_get_value() * self.opponent_factor
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

    def __fn_get_visits(self):
        return self.visits

    def __fn_get_value(self):
        return self.val

    def __fn_get_children_nodes(self):
        return self.children_nodes

