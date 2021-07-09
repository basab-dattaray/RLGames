import numpy as np


def policy_mgt(fn_get_counts):

    def fn_get_policy(state, do_random_selection=True, _test_data = None):
        def fn_policy_random_selection(fn_get_counts):
            counts = fn_get_counts(state)
            counts_sum = float(sum(counts))
            if counts_sum == 0:
                probs = [1 / len(counts)] * len(counts)
                return probs
            else:
                probs = [x / counts_sum for x in counts]
                return probs

        # def fn_policy_specific_selection(counts):
        #     best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
        #     the_best_action = np.random.choice(best_actions)
        #     probs = [0] * len(counts)
        #     probs[the_best_action] = 1
        #     return probs

        # if fn_get_counts is not None:
        #     counts = fn_get_counts(state)
        # else:
        #     counts = _test_data

        # if do_random_selection:
        #     return fn_policy_specific_selection(counts)
        # else:
        return fn_policy_random_selection(fn_get_counts)

    return fn_get_policy

# if __name__ == '__main__': # test
#     fn_get_policies = policy_mgt(fn_get_counts=None)
#
#     results_fn_mcts_probability_select_one_win = fn_get_policies(None, 0, [3, 1, -4, 3])
#     assert(results_fn_mcts_probability_select_one_win == [1, 0, 0, 0] or results_fn_mcts_probability_select_one_win == [0, 0, 0, 1])
#
#     results_fn_mcts_probability_spread_out__equal_counts = fn_get_policies(None, 1, [0, 0, 0, 0])
#     assert(results_fn_mcts_probability_spread_out__equal_counts == [.25, .25, .25, .25])
#
#     r2_fn_mcts_probability_spread_out = fn_get_policies(None, 1, [1, 3, 4, 2])
#     assert(r2_fn_mcts_probability_spread_out == [.1, .3, .4, .2])
#
#     pass
#
