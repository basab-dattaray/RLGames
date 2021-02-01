
from ws.RLEnvironments.gridworld.grid_board.display_mgt import display_mgt
from ws.RLUtils.algo_lib.policy_based.qtable_mgt import qtable_mgt

def impl_mgt(app_info):
    _env = app_info.ENV

    display_mgr = app_info.ENV.display_mgr

    _epsilon = app_info.EPSILON

    _env_config = _env.fn_get_config()
    _discount_factor = app_info.DISCOUNT_FACTOR
    _learning_rate = app_info.LEARNING_RATE

    app_fn_display_info = _env_config.DISPLAY
    _width = app_fn_display_info['WIDTH']
    _height = app_fn_display_info['HEIGHT']
    _board_blockers = app_fn_display_info['BOARD_BLOCKERS']
    _board_goal = app_fn_display_info['BOARD_GOAL']

    fn_get_qval, fn_set_qval, fn_get_q_actions, fn_get_max_q_actions = qtable_mgt()

    def _fn_update_knowledge(state, action, reward, next_state, next_action):
        current_q = fn_get_qval(state, action)
        next_state_q = fn_get_qval(next_state, next_action)
        new_q = (current_q + app_info.LEARNING_RATE *
                 (reward + app_info.DISCOUNT_FACTOR * next_state_q - current_q))
        fn_set_qval(state, action, new_q)

    def fn_bind_fn_display_actions(acton_dictionary):
        display_mgr.fn_init(acton_dictionary)

    def fn_run_sarsa():
        episode_num = 0
        while True:
            episode_num += 1
            episode_status = _fn_run_episode(display_mgr.fn_move_cursor)
            print('episode number: {}   status = {}'.format(episode_num, episode_status))

    def _fn_run_episode(fn_move_cursor):
        new_state = None

        state = _env.fn_reset_env()
        action = fn_get_max_q_actions(state, app_info.EPSILON)
        display_mgr.fn_update_ui(state, fn_get_q_actions(state))
        continue_running  = True
        while continue_running:
            new_state, reward, done, _ = _env.fn_take_step(action)
            continue_running = reward == 0
            if fn_move_cursor is not None:
                fn_move_cursor(state, new_state)

            new_action = fn_get_max_q_actions(new_state, app_info.EPSILON)
            _fn_update_knowledge(state, action, reward, new_state, new_action)
            display_mgr.fn_update_ui(state, fn_get_q_actions(state))

            action = new_action
            state = new_state
        if fn_move_cursor is not None:
            fn_move_cursor(new_state)

        return continue_running

    return fn_bind_fn_display_actions, fn_run_sarsa
