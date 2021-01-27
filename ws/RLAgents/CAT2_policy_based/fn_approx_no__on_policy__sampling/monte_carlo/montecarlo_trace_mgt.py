import numpy as np

from ws.RLUtils.algo_lib.planning.value_table_mgt import value_table_mgt
from ws.RLUtils.common.misc_functions import arg_max

def montecarlo_trace_mgt(env, app_info):
    _env = env

    fn_set_value_table_item, fn_get_value_table_item, fn_set_value_table, fn_get_value_table, \
    fn_value_table_possible_actions_given_state, _, _ = value_table_mgt(
        app_info,
        goal_coordinates=None, width=None, height=None, board_goal=None
    )

    _interaction_trace = []

    def fnTraceInteraction(state, reward, status):
        nonlocal _interaction_trace
        _interaction_trace.append([state, reward, status])

    def fnClearTrace():
        _interaction_trace.clear()

    def fnGetEpsilonGreedyAction(state):
        rn = np.random.rand()
        if rn < app_info.EPSILON:
            selected_action = np.random.choice(len(env.fn_get_allowed_moves()))
        else:
            actions = fn_value_table_possible_actions_given_state(state)
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
                G_t = app_info.DISCOUNT_FACTOR * (reward + G_t)
                value = fn_get_value_table_item(state)
                new_val = (value + app_info.LEARNING_RATE * (G_t - value))
                fn_set_value_table_item(state, new_val)
                continue
        val_table = fn_get_value_table()
        return val_table

    return fnClearTrace, fnGetEpsilonGreedyAction, fnTraceInteraction, fnUpdateValueTableFromTrace
