from collections import namedtuple

from ws.RLAgents.self_play.alpha_zero.search.cache_mgt import cache_mgt
from ws.RLAgents.self_play.alpha_zero.search.policy_mgt import policy_mgt

from ws.RLAgents.self_play.alpha_zero.search.recursive.state_visit_mgt import state_visit_mgt
from ws.RLAgents.self_play.alpha_zero.search.recursive.search_helper import search_helper


def mcts_r_mgr(
    args,
    game_mgr,
    neural_net_mgr,
    playground_mgt,
    num_mcts_simulations,
    cpuct_exploration_exploitation_factor,
    max_num_actions
):

    search_utils = search_helper(
        args,
        game_mgr,
        neural_net_mgr
    )

    def fn_get_mcts_counts(state):
        for i in range(num_mcts_simulations):
            fn_search(state)
        state_key = game_mgr.fn_get_state_key(state)
        visit_counts = search_utils.fn_get_visit_counts(state_key)
        return visit_counts

    def fn_search(state):
        state_key = game_mgr.fn_get_state_key(state)

        state_results = search_utils.fn_get_real_state_value(state)
        if state_results != 0:
            return - state_results

        prediction_info, is_new_prediction = search_utils.fn_get_predicted_based_state_value(state)
        if is_new_prediction:
            return - prediction_info['state_val']

        # select best action at this non terminal state
        best_action = search_utils.fn_get_best_ucb_action(
            state_key
        )

        next_state = game_mgr.fn_get_next_state(state, 1, best_action)
        next_state_canonical = game_mgr.fn_get_canonical_form(next_state, -1)

        state_val = fn_search(next_state_canonical)

        search_utils.fn_update_state_during_backprop(state_key, best_action, state_val)

        return -state_val

    mcts_mgr = namedtuple('_', ['fn_get_policy'])

    fn_get_policy = policy_mgt(fn_get_mcts_counts)
    mcts_mgr.fn_get_policy = fn_get_policy

    return mcts_mgr

