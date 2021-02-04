from collections import namedtuple

import gym
import numpy


def env_mgt(name, strategy= None):

    error_message = None
    try:
        _env = gym.make(name)
        _state_size = _env.observation_space.shape[0]
        _action_size = _env.action_space.n

    except Exception as x:
        error_message = x


    def fn_reset_env():
        state = _env.reset()
        state = numpy.reshape(state, (1, _state_size))
        return state

    def fn_take_step(action):
        next_state, reward, done, info = _env.step(action)
        next_state = numpy.reshape(next_state, (1, _state_size))
        return next_state, reward, done, info

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
        'ERROR_MESSAGE'
    ])

    ret_obj.fn_reset_env = fn_reset_env
    ret_obj.fn_take_step = fn_take_step
    ret_obj.fn_render = fn_render
    ret_obj.fn_get_state_size = fn_get_state_size
    ret_obj.fn_get_action_size = fn_get_action_size
    ret_obj.fn_close = fn_close
    ret_obj.ERROR_MESSAGE = error_message
    return ret_obj
