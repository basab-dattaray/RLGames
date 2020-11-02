import logging
import math

import numpy as np

from ws.RLAgents.self_play.alpha_zero.search.mcts_probability_mgt import mcts_probability_mgt

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    def __init__(self, game, nnet, args,
                 fn_get_next_state,
                 fn_get_canonical_form,
                 fn_predict_action_probablities,
                 fn_get_valid_actions,
                 fn_terminal_state_status,
                 num_mcts_simulations,
                 explore_exploit_ratio,
                 max_num_actions
                 ):
        # self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for state,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge state,a was visited
        self.Ns = {}  # stores #times board_pieces state was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.fn_get_game_progress_status ended for board_pieces state
        self.Vs = {}  # stores game.fn_get_valid_moves for board_pieces state

        self.fn_get_action_probabilities = mcts_probability_mgt(self.fn_init_mcts, self.fn_get_mcts_count)
        self.fn_get_valid_actions = fn_get_valid_actions
        self.fn_terminal_state_status = fn_terminal_state_status

        self.fn_get_next_state = fn_get_next_state
        self.fn_get_canonical_form = fn_get_canonical_form
        self.fn_predict_action_probablities = fn_predict_action_probablities

        self.num_mcts_simulations = num_mcts_simulations
        self.explore_exploit_ratio = explore_exploit_ratio
        self.max_num_actions = max_num_actions


    def fn_get_mcts_count(self, state):
        for i in range(self.args.num_of_mc_simulations):
            self.search(state)

        s = self.game.fn_get_state_key(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.max_num_actions)]
        return counts

    def fn_init_mcts(self, canonical_board):
        return None

    def search(self, state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current state
        """

        state_key = self.game.fn_get_state_key(state)

        # ROLLOUT 1 - actual result
        if state_key not in self.Es:
            self.Es[state_key] = self.fn_terminal_state_status(state)
        if self.Es[state_key] != 0:
            # terminal node
            return -self.Es[state_key]

        # ROLLOUT 2 - uses prediction
        if state_key not in self.Ps:
            # leaf node
            self.Ps[state_key], v = self.nnet.predict(state)
            valid_actions = self.fn_get_valid_actions(state)
            self.Ps[state_key] = self.Ps[state_key] * valid_actions  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[state_key])
            if sum_Ps_s > 0:
                self.Ps[state_key] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[state_key] = self.Ps[state_key] + valid_actions
                self.Ps[state_key] /= np.sum(self.Ps[state_key])

            self.Vs[state_key] = valid_actions
            self.Ns[state_key] = 0
            return -v

        # SELECTION - node already visited so find next best node in the subtree
        valid_actions = self.Vs[state_key]
        best_action = self.fn_get_best_action(state_key, valid_actions)
        next_state, next_player = self.fn_get_next_state(state, 1, best_action)
        next_state_canonical = self.fn_get_canonical_form(next_state, next_player)

        # EXPANSION
        v = self.search(next_state_canonical)

        # BACKPROP
        if (state_key, best_action) in self.Qsa: # UPDATE EXISTING
            self.Qsa[(state_key, best_action)] = (self.Nsa[(state_key, best_action)] * self.Qsa[(state_key, best_action)] + v) / (self.Nsa[(state_key, best_action)] + 1)
            self.Nsa[(state_key, best_action)] += 1

        else: # UPDATE FIRST TIME
            self.Qsa[(state_key, best_action)] = v
            self.Nsa[(state_key, best_action)] = 1

        self.Ns[state_key] += 1
        return -v

    def fn_get_best_action(self, state, valids):
        cur_best = -float('inf')
        best_act = -1
        # pick the action with the highest upper confidence bound
        for a in range(self.max_num_actions):
            if valids[a]:
                if (state, a) in self.Qsa:
                    u = self.Qsa[(state, a)] + self.args.cpuct_exploration_exploitation_factor * self.Ps[state][a] * math.sqrt(
                        self.Ns[state]) / (
                                1 + self.Nsa[(state, a)])
                else:
                    u = self.args.cpuct_exploration_exploitation_factor * self.Ps[state][a] * math.sqrt(
                        self.Ns[state] + EPS)  # Q = 0 ?
                    # u = 0

                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        return a
