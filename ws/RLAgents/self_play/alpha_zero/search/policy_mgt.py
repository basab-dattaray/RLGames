import numpy as np


def policy_mgt(fn_get_counts):

    def fn_get_policy(state, do_spread_probabilities=1, _test_data = None):
        def fn_mcts_probability_spread_out(counts):
            counts_sum = float(sum(counts))
            if counts_sum == 0:
                probs = [1 / len(counts)] * len(counts)
                return probs
            else:
                probs = [x / counts_sum for x in counts]
                return probs


        def fn_mcts_probability_select_one_win(counts):
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        if fn_get_counts is not None:
            counts = fn_get_counts(state)
        else:
            counts = _test_data

        if do_spread_probabilities == 0:
            return fn_mcts_probability_select_one_win(counts)
        else:
            return fn_mcts_probability_spread_out(counts)

    return fn_get_policy

if __name__ == '__main__': # test
    fn_get_action_probs = policy_mgt(fn_get_counts=None)

    results_fn_mcts_probability_select_one_win = fn_get_action_probs(None, 0, [3, 1, -4, 3])
    assert(results_fn_mcts_probability_select_one_win == [1, 0, 0, 0] or results_fn_mcts_probability_select_one_win == [0, 0, 0, 1])

    results_fn_mcts_probability_spread_out__equal_counts = fn_get_action_probs(None, 1, [0, 0, 0, 0])
    assert(results_fn_mcts_probability_spread_out__equal_counts == [.25, .25, .25, .25])

    r2_fn_mcts_probability_spread_out = fn_get_action_probs(None, 1, [1, 3, 4, 2])
    assert(r2_fn_mcts_probability_spread_out == [.1, .3, .4, .2])

    pass

