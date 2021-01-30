from ws.RLEnvironments.gridworld.grid_board.display_mgt import display_mgt
from ws.RLUtils.algo_lib.policy_based.qtable_mgt import qtable_mgt


def impl_mgt(app_info):
    _display_mgr = app_info.ENV.display_mgr

    fn_get_qval, fn_set_qval, fn_get_q_actions, fn_get_max_q_actions = qtable_mgt()

    def fnUpdateKnowledge(state, action, reward, next_state):
        current_q = fn_get_qval(state, action)
        new_q = reward + app_info.DISCOUNT_FACTOR * max(fn_get_q_actions(next_state))

        new_val = current_q + app_info.LEARNING_RATE * (new_q - current_q)
        fn_set_qval(state, action, new_val)

    def fn_bind_fn_display_actions(acton_dictionary):
        _display_mgr.fn_init(acton_dictionary)

    def fnQLearn():
        episode_num = 0
        while True:
            episode_num += 1
            episode_status = runEpisode()
            print('episode number: {}   status = {}'.format(episode_num, episode_status))

    def runEpisode():
        state = app_info.ENV.fn_reset_env()
        # fn_update_ui(display_mgr.fn_show_qvalue, state)
        _display_mgr.fn_update_ui(state, fn_get_q_actions(state))

        continue_running = True
        while continue_running:

            action = fn_get_max_q_actions(state, app_info.EPSILON)

            new_state, reward, _, _ = app_info.ENV.fn_take_step(action)

            fnUpdateKnowledge(state, action, reward, new_state)
            continue_running = reward == 0

            # fn_update_ui(display_mgr.fn_show_qvalue, state)
            _display_mgr.fn_update_ui(state, fn_get_q_actions(state))

            if _display_mgr.fn_move_cursor is not None:
                _display_mgr.fn_move_cursor(state, new_state)

            state = new_state

        if _display_mgr.fn_move_cursor is not None:
            _display_mgr.fn_move_cursor(state)

        return continue_running


    # def fn_update_ui(fn_show_qvalue, state):
    #     if fn_show_qvalue is not None:
    #         q_actions = fn_get_q_actions(state)
    #         fn_show_qvalue(state, q_actions)

    return fn_bind_fn_display_actions, fnQLearn