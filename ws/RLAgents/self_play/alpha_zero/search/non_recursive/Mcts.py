# PURPOSE: to reduce cost for taking the near optimal action (edge) when there are many possible actions
# The number of num_mcts_simulations determines how thorough a search you want.
# Remember that the higher the num_mcts_simulations, the better estimate because there will be more monte carlo rollouts for better estimation.
# A explore_exploit_ratio tells the search to put higher emphasis on exploration relative to exploitation.
# The more evolved board_pieces states (nodes) will have fewer allowable actions (edges)
import os


from .Node import Node
from .state_cache_mgt import state_cache_mgt
from .rollout_mgt import rollout_mgt
from ..mcts_probability_mgt import mcts_probability_mgt

MTCS_RESULTS_FILE_NAME = 'mtcs_results.pkl'
CACHE_RESULTS = False

class Mcts():

    def __init__(self,
                 fn_get_state_key,
                 fn_get_next_state,
                 fn_get_canonical_form,
                 fn_predict_action_probablities,
                 fn_get_valid_actions,
                 fn_terminal_state_status,
                 num_mcts_simulations,
                 explore_exploit_ratio,
                 max_num_actions
                 ):

        self.app_path = os.getcwd()

        self.num_mcts_simulations = num_mcts_simulations
        self.max_num_actions = max_num_actions
        self.explore_exploit_ratio = explore_exploit_ratio

        self.root_node = None

        self.fn_get_state_key = fn_get_state_key,
        self.fn_get_next_state = fn_get_next_state,
        self.fn_get_canonical_form = fn_get_canonical_form,
        self.fn_predict_action_probablities = fn_predict_action_probablities
        self.fn_get_valid_actions = fn_get_valid_actions
        self.fn_terminal_state_status = fn_terminal_state_status

        self.state_cache = None
        self.fn_get_action_probabilities = mcts_probability_mgt(self.fn_init_mcts, self.fn_get_mcts_counts)
        self.fn_rollout = rollout_mgt(self.state_cache, self.fn_predict_action_probablities, self.fn_terminal_state_status,
                    fn_get_next_state, fn_get_canonical_form,
                    multirun=False)

    def fn_execute_monte_carlo_tree_search(self, state):
        if self.root_node is None:

            self.root_node = Node(
                self.state_cache.fn_get_valid_normalized_action_probabilities,
                self.max_num_actions,
                self.explore_exploit_ratio,

                parent_action=-1,
                val=0.0,
                parent_node=None,
                state= state
            )

        selected_node = self.root_node.fn_select_from_available_leaf_nodes()

        if selected_node.fn_is_already_visited():
            selected_node = selected_node.fn_expand_node()
            if selected_node is None:
                return None
            pass

        score, terminal_state = self.fn_rollout(selected_node.state)

        # score, terminal_state = selected_node.fn_rollout()

        # if terminal_state: print(f'*** score={score}')
        value = selected_node.fn_back_propagate(score)
        return value

    def __fn_reset_mcts(self):
        self.root_node = None

    def fn_get_mcts_counts(self, state):
        def __fn_get_counts():
            childrenNodes = self.root_node.children_nodes
            counts = [0] * self.max_num_actions
            for key, val in childrenNodes.items():
                index = int(key)
                counts[index] = val.visits
            return counts

        for i in range(self.num_mcts_simulations):
            self.fn_execute_monte_carlo_tree_search(state)
        counts = __fn_get_counts()
        return counts

    def fn_init_mcts(self, state):
        self.state_cache = state_cache_mgt(self.fn_get_valid_actions, self.fn_predict_action_probablities, state)
        self.__fn_reset_mcts()
        return True