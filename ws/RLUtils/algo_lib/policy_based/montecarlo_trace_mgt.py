import numpy as np

from ws.RLUtils.algo_lib.planning.values_repo_mgt import values_repo_mgt
from ws.RLUtils.common.misc_functions import arg_max

def montecarlo_trace_mgt(env, epsilon, discount_factor, learning_rate):
    # config = env.fn_get_config()

    _stateparts_repo = values_repo_mgt(
        env
    )

    _interaction_trace = []

    def fnTraceInteraction(state, reward, status):
        nonlocal _interaction_trace
        _interaction_trace.append([state, reward, status])

    def fnClearTrace():
        _interaction_trace.clear()

    def fnGetEpsilonGreedyAction(state):
        rn = np.random.rand()
        if rn < epsilon:
            selected_action = np.random.choice(len(env.fn_get_allowed_moves()))
        else:
            actions = _stateparts_repo.fn_get_statepart_actions(state)
            selected_action = arg_max(actions)
        return selected_action

    def fnUpdateValueTableFromTrace():
        val_table = updateValueTableFromTrace()
        return val_table

    def updateValueTableFromTrace():
        G_t = 0
        visit_state = []
        for trace in reversed(_interaction_trace):
            state = trace[0]
            if state not in visit_state:
                visit_state.append(state)
                reward = trace[1]
                G_t = discount_factor * (reward + G_t)
                value = _stateparts_repo.fn_get_statepart_value(state)
                new_val = (value + learning_rate * (G_t - value))
                _stateparts_repo.fn_set_statepart_value(state, new_val)
                continue
        val_table = _stateparts_repo.fn_get_all_statepart_values()
        return val_table

    return fnClearTrace, fnGetEpsilonGreedyAction, fnTraceInteraction, fnUpdateValueTableFromTrace
