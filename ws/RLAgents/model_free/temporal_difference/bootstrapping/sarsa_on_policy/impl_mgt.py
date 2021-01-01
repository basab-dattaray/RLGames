from ws.RLInterfaces.PARAM_KEY_NAMES import OBJ_EPISODE, LEARNING_RATE, DISCOUNT_FACTOR
from ws.RLEnvironments.gridworld.grid_board.Display import Display
from ws.RLEnvironments.gridworld.logic.qtable_mgt import qtable_mgt


# from .details_mgt import details_mgt

def impl_mgt(env, app_info):
    _env = env

    _fn_display_controller = Display(app_info)

    _epsilon = app_info["EPSILON"]
    _discount_factor = app_info["DISCOUNT_FACTOR"]
    _learning_rate = app_info["LEARNING_RATE"]

    app_fn_display_info = app_info["display"]
    _width = app_fn_display_info["WIDTH"]
    _height = app_fn_display_info["HEIGHT"]
    _board_blockers = app_fn_display_info["BOARD_BLOCKERS"]
    _board_goal = app_fn_display_info["BOARD_GOAL"]
    # fn_get_q_actions, fnUpdateKnowledge, fn_get_max_q_actions = details_mgt(app_info)

    fn_get_qval, fn_set_qval, fn_get_q_actions, fn_get_max_q_actions = qtable_mgt()

    def fnUpdateKnowledge(state, action, reward, next_state, next_action):
        current_q = fn_get_qval(state, action)
        next_state_q = fn_get_qval(next_state, next_action)
        new_q = (current_q + app_info[LEARNING_RATE] *
                 (reward + app_info[DISCOUNT_FACTOR] * next_state_q - current_q))
        fn_set_qval(state, action, new_q)

    def fn_bind_fn_display_actions(acton_dictionary):
        _fn_display_controller.fnInit(acton_dictionary)

    def fnRunSarsa():
        episode_num = 0
        while True:
            episode_num += 1
            episode_status = runEpisode(_fn_display_controller.fn_move_cursor, _fn_display_controller.fn_show_qvalue)
            print('episode number: {}   status = {}'.format(episode_num, episode_status))

    def runEpisode(fn_move_cursor, fn_show_qvalue):
        new_state = None

        state = _env.fnReset()
        action = fn_get_max_q_actions(state, app_info["EPSILON"])

        episode = app_info[OBJ_EPISODE]
        while episode.fn_should_episode_continue():
            new_state, reward, episode_status, _ = _env.fnStep(action)

            # _env[envMgr__fnSetState](new_state)
            if fn_show_qvalue is not None:
                q_actions = fn_get_q_actions(state)
                fn_show_qvalue(state, q_actions)
            if fn_move_cursor is not None:
                fn_move_cursor(state, new_state)

            new_action = fn_get_max_q_actions(new_state, app_info["EPSILON"])
            fnUpdateKnowledge(state, action, reward, new_state, new_action)

            action = new_action
            state = new_state

        if fn_move_cursor is not None:
            fn_move_cursor(new_state)

        return episode.fn_get_episode_status()

    return fn_bind_fn_display_actions, fnRunSarsa
