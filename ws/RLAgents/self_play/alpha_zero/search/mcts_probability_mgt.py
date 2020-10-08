import numpy as np


def mcts_adapter_mgt(fn_run_simulations, fn_get_counts, num_simulations):

    def fn_getActionProb(canonicalBoard, temp=1):
        temp = 0

        fn_run_simulations(canonicalBoard, num_simulations)

        counts = fn_get_counts(canonicalBoard)

        if temp == 0:
            return fn_mcts_probability_select_one_win(counts)

        return fn_mcts_probability_spread_out(counts)


    def fn_mcts_probability_spread_out(counts):
        counts = [x ** (1. / 1) for x in counts]
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
