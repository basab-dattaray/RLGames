from ws.RLAgents.B_ValueBased.Bootstrapping.qtable_mgt import qtable_mgt

def impl_mgt(app_info):
    _env = app_info.ENV

    Display = app_info.ENV.Display
    fn_get_qval, fn_set_qval, fn_get_q_actions, fn_get_max_q_actions = qtable_mgt()

    def _fn_update_knowledge(state, action, reward, next_state, next_action):
        current_q = fn_get_qval(state, action)
        next_state_q = fn_get_qval(next_state, next_action)
        new_q = (current_q + app_info.LEARNING_RATE *
                 (reward + app_info.DISCOUNT_FACTOR * next_state_q - current_q))
        fn_set_qval(state, action, new_q)

    def fn_run_sarsa():
        episode_num = 0
        while True:
            episode_num += 1
            episode_status = _fn_run_episode(Display.fn_move_cursor)
            print('episode number: {}   status = {}'.format(episode_num, episode_status))
            if 'TEST_MODE' in app_info:
                if app_info.TEST_MODE: # ONLY 1 episode needed
                    break
        pass

    def _fn_run_episode(fn_move_cursor):
        new_state = None

        state = _env.fn_reset_env()
        action = fn_get_max_q_actions(state, app_info.EPSILON)
        Display.fn_update_ui(state, fn_get_q_actions(state))
        continue_running  = True
        while continue_running:
            new_state, reward, done, _ = _env.fn_take_step(action)
            continue_running = reward == 0
            if fn_move_cursor is not None:
                fn_move_cursor(state, new_state)

            new_action = fn_get_max_q_actions(new_state, app_info.EPSILON)
            _fn_update_knowledge(state, action, reward, new_state, new_action)
            Display.fn_update_ui(state, fn_get_q_actions(state))

            action = new_action
            state = new_state
        if fn_move_cursor is not None:
            fn_move_cursor(new_state)

        return continue_running

    return fn_run_sarsa
