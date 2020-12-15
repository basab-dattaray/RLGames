from collections import namedtuple

from ws.RLAgents.self_play.alpha_zero.search.cache_mgt import cache_mgt
from ws.RLAgents.self_play.alpha_zero.search.policy_mgt import policy_mgt

from ws.RLAgents.self_play.alpha_zero.search.recursive.state_visit_mgt import state_visit_mgt
from ws.RLAgents.self_play.alpha_zero.search.recursive.search_helper import search_helper, create_normalized_predictor


def mcts_r_mgr(
    args,
    game_mgr,
    neural_net_mgr,
    playground_mgt,
    num_mcts_simulations,
    explore_exploit_ratio,
    max_num_actions
):
    fn_terminal_value = lambda pieces: game_mgr.fn_get_game_progress_status(pieces, 1)
    fn_get_valid_actions = lambda board: game_mgr.fn_get_valid_moves(board, 1)
    fn_get_prediction_info_3 = create_normalized_predictor (neural_net_mgr.predict, fn_get_valid_actions)

    state_visits = state_visit_mgt()

    cache_mgr = cache_mgt()

    search_help = search_helper(
        args,
        cache_mgr.state_action_qval,
        cache_mgr.state_policy,
        state_visits
    )

    def fn_get_mcts_counts(state):
        for i in range(num_mcts_simulations):
            fn_search(state)

        s = game_mgr.fn_get_state_key(state)
        counts = [state_visits.fn_get_child_state_visits((s, a)) if state_visits.fn_does_child_state_visits_exist((s, a)) else 0 for a in range(max_num_actions)]
        sum_counts = sum(counts)
        return counts


    fn_get_policy = policy_mgt(fn_get_mcts_counts)

    def fn_search(state):
        state_key = game_mgr.fn_get_state_key(state)

        # ROLLOUT 1 - actual result
        if not cache_mgr.state_results.fn_does_key_exist(state_key):
            cache_mgr.state_results.fn_set_data(state_key, fn_terminal_value(state))
        if cache_mgr.state_results.fn_get_data(state_key) != 0:
            # terminal node
            return -cache_mgr.state_results.fn_get_data(state_key)

        # ROLLOUT 2 - uses prediction
        if not cache_mgr.state_info.fn_does_key_exist(state_key):
            # leaf node
            policy, state_val, valid_actions = fn_get_prediction_info_3(state)
            if valid_actions is None:
                return -state_val
            state_info = {
                'policy': policy,
                'state_val': state_val,
                'valid_actions': valid_actions
            }
            cache_mgr.state_info.fn_set_data(state_key, state_info)
            cache_mgr.state_policy.fn_set_data(state_key, policy)

            cache_mgr.state_valid_moves.fn_set_data(state_key, valid_actions)

            # Ns[state_key] = 0
            state_visits.fn_set_state_visits(state_key, 0)

            return -state_val

        # SELECTION - node already visited so find next best node in the subtree

        best_action = search_help.fn_get_best_ucb_action(cache_mgr, state_key, max_num_actions, explore_exploit_ratio)
        next_state = game_mgr.fn_get_next_state(state, 1, best_action)
        next_state_canonical = game_mgr.fn_get_canonical_form(next_state, -1)

        # BACKPROP
        state_val = fn_search(next_state_canonical)

        search_help.fn_update_state_during_backprop(state_key, best_action, state_val)

        return -state_val

    mcts_mgr = namedtuple('_', ['fn_get_policy'])
    mcts_mgr.fn_get_policy = fn_get_policy

    return mcts_mgr

# Qsa = {}  # stores Q values for state_key,action (as defined in the paper)

# Ps = {}  # stores initial policy (returned by neural net)
# Es = {}  # stores game.fn_get_game_progress_status ended for board_pieces state_key
# Vs = {}  # stores game.fn_get_valid_moves for board_pieces state_key
