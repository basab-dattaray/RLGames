from ws.RLEnvironments.gridworld.grid_board.display_mgt import display_mgt
from ws.RLAgents.logic.qtable_mgt import qtable_mgt


def impl_mgt(env, app_info):
    _env = env
    _fn_display_controller = display_mgt(app_info)

    fn_get_qval, fn_set_qval, fn_get_q_actions, fn_get_max_q_actions = qtable_mgt()

    def fnUpdateKnowledge(state, action, reward, next_state):
        current_q = fn_get_qval(state, action)
        new_q = reward + app_info['DISCOUNT_FACTOR'] * max(fn_get_q_actions(next_state))

        new_val = current_q + app_info['LEARNING_RATE'] * (new_q - current_q)
        fn_set_qval(state, action, new_val)

    def fn_bind_fn_display_actions(acton_dictionary):
        _fn_display_controller.fn_init(acton_dictionary)

    def fnQLearn():
        episode_num = 0
        while True:
            episode_num += 1
            episode_status = runEpisode()
            print('episode number: {}   status = {}'.format(episode_num, episode_status))

    def runEpisode():
        state = _env.fn_reset_env()
        update_ui(_fn_display_controller.fn_show_qvalue, state)

        continue_running = True
        while continue_running:

            action = fn_get_max_q_actions(state, app_info['EPSILON'])

            new_state, reward, _, _ = _env.fn_take_step(action)

            fnUpdateKnowledge(state, action, reward, new_state)
            continue_running = reward == 0

            update_ui(_fn_display_controller.fn_show_qvalue, state)

            if _fn_display_controller.fn_move_cursor is not None:
                _fn_display_controller.fn_move_cursor(state, new_state)

            state = new_state

        if _fn_display_controller.fn_move_cursor is not None:
            _fn_display_controller.fn_move_cursor(state)

        return continue_running

    def update_ui(fn_show_qvalue, state):
        if fn_show_qvalue is not None:
            q_actions = fn_get_q_actions(state)
            fn_show_qvalue(state, q_actions)

    return fn_bind_fn_display_actions, fnQLearn
