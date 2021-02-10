from ws.RLAgents.B_ValueBased.Bootstrapping.qtable_mgt import qtable_mgt


def impl_mgt(app_info):
    Display = app_info.ENV.Display

    fn_get_qval, fn_set_qval, fn_get_q_actions, fn_get_max_q_actions = qtable_mgt()
    _test_mode = False

    def _fn_update_knowledge(state, action, reward, next_state):
        current_q = fn_get_qval(state, action)
        new_q = reward + app_info.DISCOUNT_FACTOR * max(fn_get_q_actions(next_state))

        new_val = current_q + app_info.LEARNING_RATE * (new_q - current_q)
        fn_set_qval(state, action, new_val)

    def fn_q_learn():
        episode_num = 0
        while True:
            episode_num += 1
            episode_status = _fn_run_episode()
            print('episode number: {}   status = {}'.format(episode_num, episode_status))
            if 'TEST_MODE' in app_info:
                if app_info.TEST_MODE: # ONLY 1 episode needed
                    break

    def _fn_run_episode():
        state = app_info.ENV.fn_reset_env()
        # fn_update_qvalue(Display.fn_show_qvalue, state)
        Display.fn_update_qvalue(state, fn_get_q_actions(state))

        continue_running = True
        while continue_running:

            action = fn_get_max_q_actions(state, app_info.EPSILON)

            new_state, reward, _, _ = app_info.ENV.fn_take_step(action)

            _fn_update_knowledge(state, action, reward, new_state)
            continue_running = reward == 0

            # fn_update_qvalue(Display.fn_show_qvalue, state)
            Display.fn_update_qvalue(state, fn_get_q_actions(state))

            if Display.fn_move_cursor is not None:
                Display.fn_move_cursor(state, new_state)

            state = new_state

        if Display.fn_move_cursor is not None:
            Display.fn_move_cursor(state)

        return continue_running

    return fn_q_learn
