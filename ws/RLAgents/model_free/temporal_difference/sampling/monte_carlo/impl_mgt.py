from ws.RLInterfaces.PARAM_KEY_NAMES import OBJ_EPISODE, NUM_EPISODES
from ws.RLEnvironments.gridworld.grid_board.Display import Display

from .details_mgt import details_mgt


def impl_mgt(env, app_info):
    _env = env

    _fn_display_controller = Display(app_info)
    fnClearTrace, fnGetEpsilonGreedyAction, fnTraceInteraction, fnUpdateValueTableFromTrace = details_mgt(_env,
                                                                                                          app_info)

    def fn_bind_fn_display_actions(acton_dictionary):
        _fn_display_controller.fnInit(acton_dictionary)

    def fnRunMonteCarlo():
        fnClearTrace()
        for episode in range(app_info[NUM_EPISODES]):
            value_table, episode_status = runEpisode(_fn_display_controller.fn_move_cursor)
            if _fn_display_controller.fn_show_state_values is not None:
                _fn_display_controller.fn_show_state_values(value_table)

    def runEpisode(fn_move_cursor):
        new_state = None
        state = _env.fnReset()
        action = fnGetEpsilonGreedyAction(state)

        episode = app_info[OBJ_EPISODE]
        while episode.fn_should_episode_continue():
            new_state, reward, episode_status, _ = _env.fnStep(action)

            fnTraceInteraction(new_state, reward, episode_status)
            if fn_move_cursor is not None:
                fn_move_cursor(state, new_state)

            action = fnGetEpsilonGreedyAction(new_state)

            state = new_state
        if fn_move_cursor is not None:
            fn_move_cursor(new_state)
        value_table = fnUpdateValueTableFromTrace()

        return value_table, episode.fn_get_episode_status()

    return fn_bind_fn_display_actions, fnRunMonteCarlo
