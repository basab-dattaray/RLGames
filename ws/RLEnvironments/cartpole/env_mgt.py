from collections import namedtuple

import gym

def env_mgt(app_info):

    _env = gym.make(app_info.ENV_NAME)
    _state_size = _env.observation_space.shape[0]
    _action_size = _env.action_space.n

    def fn_reset_env():
        return _env.reset()

    def fn_take_step(action):
        next_state, reward, done, _ = _env.step(action)
        return next_state, reward, done, None

    def fn_render():
        _env.render()

    def fn_get_state_size():
        return [_state_size][0]

    def fn_get_action_size():
        return [_action_size][0]

    def fn_close():
        _env.close()

    ret_obj = namedtuple('_', [
        'fn_reset_env',
        'fn_take_step',
        'fn_render',
        'fn_get_state_size',
        'fn_get_action_size',
        'fn_close',
    ])

    ret_obj.fn_reset_env = fn_reset_env
    ret_obj.fn_take_step = fn_take_step
    ret_obj.fn_render = fn_render
    ret_obj.fn_get_state_size = fn_get_state_size
    ret_obj.fn_get_action_size = fn_get_action_size
    ret_obj.fn_close = fn_close

    return ret_obj