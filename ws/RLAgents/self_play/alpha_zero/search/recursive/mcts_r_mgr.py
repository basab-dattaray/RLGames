from collections import namedtuple

from ws.RLAgents.self_play.alpha_zero.search.cache_mgt import cache_mgt
from ws.RLAgents.self_play.alpha_zero.search.policy_mgt import policy_mgt



# log = logging.getLogger(__name__)
from ws.RLAgents.self_play.alpha_zero.search.search_helper import search_helper


def mcts_r_mgr(
    fn_get_prediction_info,
    fn_get_state_key,
    fn_get_next_state,
    fn_get_canonical_form,
    fn_terminal_value,
    num_mcts_simulations,
    explore_exploit_ratio,
    max_num_actions
):

    # Qsa = {}  # stores Q values for state_key,action (as defined in the paper)

    # Ps = {}  # stores initial policy (returned by neural net)
    # Es = {}  # stores game.fn_get_game_progress_status ended for board_pieces state_key
    # Vs = {}  # stores game.fn_get_valid_moves for board_pieces state_key

    Nsa = {}  # stores #times edge state_key,action was visited
    Ns = {}  # stores #times board_pieces state_key was visited
    fn_get_state_visits = lambda s: Ns[s]
    fn_get_child_state_visits = lambda sa: Nsa[sa]
    fn_does_child_state_visits_exist = lambda sa: sa in Nsa

    def fn_set_state_visits(state_key, visits):
        nonlocal Ns
        Ns[state_key] = visits

    def fn_incr_state_visits(state_key):
        nonlocal Ns
        Ns[state_key] += 1

    def fn_set_child_state_visits(state_action_key, visits):
        nonlocal Nsa
        Nsa[state_action_key] = visits

    def fn_incr_child_state_visits(state_action_key):
        nonlocal Nsa
        Nsa[state_action_key] += 1

    # cache_mgr = search_cache_mgt()
    cache_mgr = cache_mgt()

    search_help = search_helper(cache_mgr.state_action_qval, cache_mgr.state_policy,
                      fn_get_state_visits,
                      fn_get_child_state_visits,
                      fn_incr_state_visits,
                      fn_set_child_state_visits,
                      fn_incr_child_state_visits
    )

    def fn_get_mcts_counts(state):
        for i in range(num_mcts_simulations):
            fn_search(state)

        s = fn_get_state_key(state)
        counts = [fn_get_child_state_visits((s, a)) if fn_does_child_state_visits_exist((s, a)) else 0 for a in range(max_num_actions)]
        return counts


    fn_get_policy = policy_mgt(fn_get_mcts_counts)

    def fn_search(state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till action leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once action leaf node is found, the neural network is called to return an
        initial policy P and action value v for the state_key. This value is propagated
        up the search path. In case the leaf node is action terminal state_key, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state_key. This is done since v is in [-1,1] and if v is the value of action
        state_key for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current state_key
        """

        state_key = fn_get_state_key(state)

        # ROLLOUT 1 - actual result
        if not cache_mgr.state_results.fn_does_key_exist(state_key):
            cache_mgr.state_results.fn_set_data(state_key, fn_terminal_value(state))
        if cache_mgr.state_results.fn_get_data(state_key) != 0:
            # terminal node
            return -cache_mgr.state_results.fn_get_data(state_key)

        # ROLLOUT 2 - uses prediction
        if not cache_mgr.state_policy.fn_does_key_exist(state_key):
            # leaf node
            policy, state_val, valid_actions = fn_get_prediction_info(state)
            cache_mgr.state_policy.fn_set_data(state_key, policy)

            cache_mgr.state_valid_moves.fn_set_data(state_key, valid_actions)

            # Ns[state_key] = 0
            fn_set_state_visits(state_key, 0)

            return -state_val

        # SELECTION - node already visited so find next best node in the subtree
        valid_actions = cache_mgr.state_valid_moves.fn_get_data(state_key)

        best_action = search_help.fn_get_best_ucb_action(state_key, valid_actions, max_num_actions, explore_exploit_ratio)
        next_state, next_player = fn_get_next_state(state, 1, best_action)
        next_state_canonical = fn_get_canonical_form(next_state, next_player)

        # BACKPROP

        state_val = fn_search(next_state_canonical)

        search_help.fn_update_state_during_backprop(state_key, best_action, state_val)

        return -state_val

    # def fn_update_state_during_backprop(state_key, action, state_val):
    #     state_action_key = (state_key, action)
    #     if cache_mgr.state_action_qval.fn_does_key_exist(state_action_key):  # UPDATE EXISTING
    #         tmp_val = (fn_get_child_state_visits(state_action_key) * cache_mgr.state_action_qval.fn_get_data(
    #             state_action_key) + state_val) / (fn_get_child_state_visits(state_action_key) + 1)
    #         cache_mgr.state_action_qval.fn_set_data(state_action_key, tmp_val)
    #
    #         # Nsa[(state_action_key)] += 1
    #         fn_incr_child_state_visits(state_action_key)
    #
    #     else:  # UPDATE FIRST TIME
    #         cache_mgr.state_action_qval.fn_set_data(state_action_key, state_val)
    #
    #         # Nsa[(state_action_key)] = 1
    #         fn_set_child_state_visits(state_action_key, 1)
    #     # Ns[state_key] += 1
    #     fn_incr_state_visits(state_key)

    mcts_mgr = namedtuple('_', ['fn_get_policy'])
    mcts_mgr.fn_get_policy = fn_get_policy

    return mcts_mgr