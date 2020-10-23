from ws.RLInterfaces.PARAM_KEY_NAMES import OBJ_EPISODE, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE
from ws.RLEnvironments.gridworld.grid_board.Display import Display
from ws.RLEnvironments.gridworld.logic.qtable_mgt import qtable_mgt


def impl_mgt(env, app_info):
    _env = env
    _fn_display_controller = Display(app_info)

    fn_get_qval, fn_set_qval, fn_get_q_actions, fn_get_max_q_actions = qtable_mgt()

    def fnUpdateKnowledge(state, action, reward, next_state):
        current_q = fn_get_qval(state, action)
        new_q = reward + app_info[DISCOUNT_FACTOR] * max(fn_get_q_actions(next_state))

        new_val = current_q + app_info[LEARNING_RATE] * (new_q - current_q)
        fn_set_qval(state, action, new_val)

    def fn_bind_fn_display_actions(acton_dictionary):
        _fn_display_controller.fnInit(acton_dictionary)

    def fnQLearn():
        episode_num = 0
        while True:
            episode_num += 1
            episode_status = runEpisode()
            print('episode number: {}   status = {}'.format(episode_num, episode_status))

    def runEpisode():
        state = _env.fnReset()
        update_ui(_fn_display_controller.fnShowQValue, state)

        episode = app_info[OBJ_EPISODE]
        while episode.fn_should_episode_continue():

            action = fn_get_max_q_actions(state, app_info[EPSILON])

            new_state, reward, episode_status, _ = _env.fnStep(action)

            fnUpdateKnowledge(state, action, reward, new_state)

            update_ui(_fn_display_controller.fnShowQValue, state)

            if _fn_display_controller.fnMoveCursor is not None:
                _fn_display_controller.fnMoveCursor(state, new_state)

            state = new_state

        if _fn_display_controller.fnMoveCursor is not None:
            _fn_display_controller.fnMoveCursor(state)

        return episode.fn_get_episode_status()

    def update_ui(fnShowQValue, state):
        if fnShowQValue is not None:
            q_actions = fn_get_q_actions(state)
            fnShowQValue(state, q_actions)

    return fn_bind_fn_display_actions, fnQLearn
