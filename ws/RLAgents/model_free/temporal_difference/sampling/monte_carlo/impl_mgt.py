from ws.RLInterfaces.PARAM_KEY_NAMES import OBJ_EPISODE, NUM_EPISODES
from ws.RLEnvironments.gridworld.grid_board.Display import Display

from .details_mgt import details_mgr


def impl_mgr(env, app_info):
    _env = env

    _display_controller = Display(app_info)
    fnClearTrace, fnGetEpsilonGreedyAction, fnTraceInteraction, fnUpdateValueTableFromTrace = details_mgr(_env,
                                                                                                          app_info)

    def fn_bind_display_actions(acton_dictionary):
        _display_controller.fnInit(acton_dictionary)

    def fnRunMonteCarlo():
        fnClearTrace()
        for episode in range(app_info[NUM_EPISODES]):
            value_table, episode_status = runEpisode(_display_controller.fnMoveCursor)
            if _display_controller.fnShowStateValues is not None:
                _display_controller.fnShowStateValues(value_table)

    def runEpisode(fnMoveCursor):
        new_state = None
        state = _env.fnReset()
        action = fnGetEpsilonGreedyAction(state)

        episode = app_info[OBJ_EPISODE]
        while episode.fn_should_episode_continue():
            new_state, reward, episode_status, _ = _env.fnStep(action)

            fnTraceInteraction(new_state, reward, episode_status)
            if fnMoveCursor is not None:
                fnMoveCursor(state, new_state)

            action = fnGetEpsilonGreedyAction(new_state)

            state = new_state
        if fnMoveCursor is not None:
            fnMoveCursor(new_state)
        value_table = fnUpdateValueTableFromTrace()

        return value_table, episode.fn_get_episode_status()

    return fn_bind_display_actions, fnRunMonteCarlo
