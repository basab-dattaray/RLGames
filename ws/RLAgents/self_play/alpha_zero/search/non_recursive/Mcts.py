# PURPOSE: to reduce cost for taking the near optimal action (edge) when there are many possible actions
# The number of num_mcts_simulations determines how thorough a search you want.
# Remember that the higher the num_mcts_simulations, the better estimate because there will be more monte carlo rollouts for better estimation.
# A explore_exploit_ratio tells the search to put higher emphasis on exploration relative to exploitation.
# The more evolved board states (nodes) will have fewer allowable actions (edges)
import os

import numpy

from .Node import Node
from .StateCache import StateCache

MTCS_RESULTS_FILE_NAME = 'mtcs_results.pkl'
CACHE_RESULTS = False

class Mcts():

    def __init__(self,fn_find_next_state, fn_predict_action_probablities, fn_get_valid_actions, fn_terminal_state_status, num_mcts_simulations, explore_exploit_ratio, max_num_actions):

        self.app_path = os.getcwd()

        self.num_mcts_simulations = num_mcts_simulations
        self.max_num_actions = max_num_actions
        self.explore_exploit_ratio = explore_exploit_ratio
        
        # self.node_visits = NodeVisitCounter()
        self.root_node = None

        self.fn_find_next_state = fn_find_next_state
        self.fn_predict_action_probablities = fn_predict_action_probablities
        self.fn_get_valid_actions = fn_get_valid_actions
        self.fn_terminal_state_status = fn_terminal_state_status

        self.state_cache = None

    def __fn_get_counts(self):
        childrenNodes = self.root_node.children_nodes
        counts = [0] * self.max_num_actions
        for key, val in childrenNodes.items():
            index = int(key)
            counts[index] = val.visits
        return counts

    def __fn_execute_monte_carlo_tree_search(self, state):
        if self.root_node is None:

            self.root_node = Node(
                self,
                self.max_num_actions,
                self.explore_exploit_ratio,

                val=0.0,
                parent_node=None,
                state= state
            )
            #! self.root_node.fn_expand_node()

        selected_node = self.root_node.fn_select_from_available_leaf_nodes()

        if selected_node.fn_is_already_visited():
            selected_node = selected_node.fn_expand_node()
            if selected_node is None:
                return None
            pass

        score, terminal_state = selected_node.fn_rollout()
        # if terminal_state: print(f'*** score={score}')
        selected_node.fn_back_propagate(score)
        pass

    def __fn_reset_mcts(self):
        self.root_node = None

    # def fn_get_action_probabilities2(self, state, temp= 1):
    #
    #     self.state_cache = StateCache(self, state)
    #
    #     self.__fn_reset_mcts()
    #
    #     counts = self.fn_get_mcts_counts(state)
    #
    #     sum_counts = numpy.sum(counts)
    #     if sum_counts == 0:
    #         return None
    #     mixed_probs = counts/sum_counts
    #     best_action = numpy.random.choice(len(mixed_probs), p=mixed_probs)
    #     probs = [0] * len(counts)
    #     probs[best_action] = 1
    #     return probs

    def fn_get_mcts_counts(self, state):
        for i in range(self.num_mcts_simulations):
            self.__fn_execute_monte_carlo_tree_search(state)
        counts = self.__fn_get_counts()
        return counts

    def fn_get_action_probabilities(self, state, temp=1):
        self.state_cache = StateCache(self, state)

        self.__fn_reset_mcts()

        # for i in range(self.num_mcts_simulations):
        #     self.__fn_execute_monte_carlo_tree_search(state)
        #
        # counts = self.__fn_get_counts()
        counts = self.fn_get_mcts_counts(state)

        if temp == 0 or True:
            bestAs = numpy.array(numpy.argwhere(counts == numpy.max(counts))).flatten()
            bestA = numpy.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs