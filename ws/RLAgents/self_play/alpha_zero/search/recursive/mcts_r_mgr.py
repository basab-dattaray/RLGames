import logging
import math
from collections import namedtuple

import numpy as np

from ws.RLAgents.self_play.alpha_zero.search.mcts_probability_mgt import mcts_probability_mgt

EPS = 1e-8
log = logging.getLogger(__name__)

def mcts_r_mgr(
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

    Qsa = {}  # stores Q values for state,a (as defined in the paper)
    Nsa = {}  # stores #times edge state,a was visited
    Ns = {}  # stores #times board_pieces state was visited
    Ps = {}  # stores initial policy (returned by neural net)

    Es = {}  # stores game.fn_get_game_progress_status ended for board_pieces state
    Vs = {}  # stores game.fn_get_valid_moves for board_pieces state

    def fn_get_mcts_counts(state):
        for i in range(num_mcts_simulations):
            search(state)

        s = fn_get_state_key(state)
        counts = [Nsa[(s, a)] if (s, a) in Nsa else 0 for a in range(max_num_actions)]
        return counts

    def fn_init_mcts(canonical_board):
        return None

    fn_predict_action_probablities = fn_predict_action_probablities
    fn_get_state_key = fn_get_state_key
    fn_get_action_probabilities = mcts_probability_mgt(fn_init_mcts, fn_get_mcts_counts)
    fn_get_valid_actions = fn_get_valid_actions
    fn_terminal_state_status = fn_terminal_state_status

    fn_get_next_state = fn_get_next_state
    fn_get_canonical_form = fn_get_canonical_form
    fn_predict_action_probablities = fn_predict_action_probablities

    num_mcts_simulations = num_mcts_simulations
    explore_exploit_ratio = explore_exploit_ratio
    max_num_actions = max_num_actions

    def search(state):
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

        state_key = fn_get_state_key(state)

        # ROLLOUT 1 - actual result
        if state_key not in Es:
            Es[state_key] = fn_terminal_state_status(state)
        if Es[state_key] != 0:
            # terminal node
            return -Es[state_key]

        # ROLLOUT 2 - uses prediction
        if state_key not in Ps:
            # leaf node
            pi, v, valid_actions = fn_get_normalized_predictions(fn_predict_action_probablities, fn_get_valid_actions, state)
            Ps[state_key] = pi
            Vs[state_key] = valid_actions
            Ns[state_key] = 0
            return -v

        # SELECTION - node already visited so find next best node in the subtree
        valid_actions = Vs[state_key]
        best_action = fn_get_best_action(state_key, valid_actions)
        next_state, next_player = fn_get_next_state(state, 1, best_action)
        next_state_canonical = fn_get_canonical_form(next_state, next_player)

        # EXPANSION
        v = search(next_state_canonical)

        # BACKPROP
        if (state_key, best_action) in Qsa: # UPDATE EXISTING
            Qsa[(state_key, best_action)] = (Nsa[(state_key, best_action)] * Qsa[(state_key, best_action)] + v) / (Nsa[(state_key, best_action)] + 1)
            Nsa[(state_key, best_action)] += 1

        else: # UPDATE FIRST TIME
            Qsa[(state_key, best_action)] = v
            Nsa[(state_key, best_action)] = 1

        Ns[state_key] += 1
        return -v

    def fn_get_normalized_predictions(fn_predict_action_probablities, fn_get_valid_actions, state):
        pi, v = fn_predict_action_probablities(state)
        valid_actions = fn_get_valid_actions(state)
        pi = pi * valid_actions  # masking invalid moves
        sum_Ps_s = np.sum(pi)
        if sum_Ps_s > 0:
            pi /= sum_Ps_s  # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable

            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
            log.error("All valid moves were masked, doing a workaround.")
            pi = pi + valid_actions
            pi /= np.sum(pi)
        return pi, v, valid_actions

    def fn_get_best_action(state, valids):
        cur_best = -float('inf')
        best_act = -1
        # pick the action with the highest upper confidence bound
        for a in range(max_num_actions):
            if valids[a]:
                if (state, a) in Qsa:
                    u = Qsa[(state, a)] + explore_exploit_ratio * Ps[state][a] * math.sqrt(
                        Ns[state]) / (
                                1 + Nsa[(state, a)])
                else:
                    u = explore_exploit_ratio * Ps[state][a] * math.sqrt(
                        Ns[state] + EPS)  # Q = 0 ?
                    # u = 0

                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        return a

    mcts_mgr = namedtuple('_', ['fn_get_action_probabilities'])
    # mcts_mgr.fn_init_mcts=fn_init_mcts
    # mcts_mgr.fn_get_mcts_counts=fn_get_mcts_counts
    mcts_mgr.fn_get_action_probabilities = fn_get_action_probabilities

    return mcts_mgr