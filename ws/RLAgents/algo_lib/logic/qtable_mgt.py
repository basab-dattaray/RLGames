from collections import defaultdict
import numpy as np

from ws.RLUtils.common.misc_functions import arg_max


def qtable_mgt():
    _q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def fn_get_qval(state, action):
        return _q_table[str(state)][action]

    def fn_set_qval(state, action, value):
        nonlocal _q_table
        _q_table[str(state)][action] = value

    def fn_get_q_actions(state):
        return _q_table[str(state)]

    def fn_get_max_q_actions(state, epsilon):
        indices = [0, 1, 2, 3]
        if np.random.rand() < epsilon:
            action = np.random.choice(indices)
            return action

        state_actions = _q_table[str(state)]

        action = arg_max(state_actions)

        return action


    return fn_get_qval , fn_set_qval, fn_get_q_actions, fn_get_max_q_actions