
from ws.RLEnvironments.gridworld.grid_board.display_mgt import display_mgt

from ws.RLUtils.algo_lib.policy_based.montecarlo_trace_mgt import montecarlo_trace_mgt


def impl_mgt(app_info):
    display_mgr = app_info.ENV.display_mgr
    fnClearTrace, fnGetEpsilonGreedyAction, fnTraceInteraction, fnUpdateValueTableFromTrace = montecarlo_trace_mgt(
        app_info.ENV,
        app_info.EPSILON,
        app_info.DISCOUNT_FACTOR,
        app_info.LEARNING_RATE,
    )

    def fn_bind_fn_display_actions(acton_dictionary):
        display_mgr.fn_init(acton_dictionary)

    def fnRunMonteCarlo():
        fnClearTrace()
        for episode in range(app_info.NUM_EPISODES):
            value_table, episode_status = runEpisode(display_mgr.fn_move_cursor)
            if display_mgr.fn_show_state_values is not None:
                display_mgr.fn_show_state_values(value_table)

    def runEpisode(fn_move_cursor):
        new_state = None
        state = app_info.ENV.fn_reset_env()
        action = fnGetEpsilonGreedyAction(state)

        continue_running = True
        while continue_running:
            new_state, reward, done, info = app_info.ENV.fn_take_step(action)
            continue_running = reward == 0
            fnTraceInteraction(new_state, reward, continue_running)
            if fn_move_cursor is not None:
                fn_move_cursor(state, new_state)

            action = fnGetEpsilonGreedyAction(new_state)

            state = new_state

        if fn_move_cursor is not None:
            fn_move_cursor(new_state)
        value_table = fnUpdateValueTableFromTrace()

        return value_table, continue_running

    return fn_bind_fn_display_actions, fnRunMonteCarlo
