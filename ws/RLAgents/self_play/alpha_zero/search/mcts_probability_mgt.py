import numpy as np


def mcts_adapter_mgt(fn_init_mcts=None, fn_get_counts=None, num_simulations=20):

    def fn_getActionProb(canonicalBoard, spread_probabilities=1):
        fn_init_mcts(canonicalBoard)
        counts = fn_get_counts(canonicalBoard)
        if spread_probabilities == 0:
            return fn_mcts_probability_select_one_win(counts)
        else:
            return fn_mcts_probability_spread_out(counts)

    def fn_mcts_probability_spread_out(counts):
        # counts = [x ** (1. / 1) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs


    def fn_mcts_probability_select_one_win(counts):
        bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
        bestA = np.random.choice(bestAs)
        probs = [0] * len(counts)
        probs[bestA] = 1
        return probs

    return fn_getActionProb
