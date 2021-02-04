import os
from collections import namedtuple

import numpy as np

# from ws.RLUtils.setup.preparation_mgt import preparation_mgt
from ws.RLUtils.monitoring.tracing.tracer import tracer
from ws.RLUtils.setup.startup_mgt import startup_mgt
from .impl_mgt import impl_mgt

def agent_mgt(caller_file):
    app_info = startup_mgt(caller_file, __file__)
    fn_load_weights = None
    if app_info.ENV is not None:
        _action_size = app_info.ENV.fn_get_action_size()
        _state_size = app_info.ENV.fn_get_state_size()
        fn_reset_env, fn_remember, fn_act, fn_replay, fn_save_weights, fn_load_weights = impl_mgt(app_info, _state_size,
                                                                                        _action_size)

    @tracer(app_info, verboscity= 4)
    def fn_change_args(change_args):
        if change_args is not None:
            for k, v in change_args.items():
                app_info[k] = v
                app_info.trace_mgr.fn_write(f'  app_info[{k}] = {v}')
        agent_mgr.APP_INFO = app_info
        return agent_mgr

    def fn_train():
        if app_info.ENV is None:
            return agent_mgr

        def fn_run_training_episodes():
            # nonlocal fn_remember
    
            num_episodes = app_info.NUM_EPISODES
            loss = []
            for episode_num in range(1, num_episodes):
                state = app_info.ENV.fn_reset_env()
                state = np.reshape(state, (1, _state_size))
                score = 0

                for step_num in range(app_info.MAX_STEPS_PER_EPISODE):
                    action = fn_act(state)
                    app_info.ENV.fn_render()
                    next_state, reward, done, _ = app_info.ENV.fn_take_step(action)
                    score += reward
                    next_state = np.reshape(next_state, (1, _state_size))
                    fn_remember(state, action, reward, next_state, done)
                    state = next_state
                    fn_replay()
    
                    if done or (step_num == app_info.MAX_STEPS_PER_EPISODE - 1) or (score >= app_info.REWARD_GOAL):
                        print("episode: {}/{}, num of steps: {}, score: {}".format(episode_num, num_episodes, step_num, score))
                        break
                loss.append(score)
                if 'TEST_MODE' in app_info:
                    if app_info.TEST_MODE:  # ONLY 1 episode needed
                        break
    
            return loss
    
        cwd = __file__.rsplit('/', 1)[0]
        model_dir = os.path.join(cwd, "models")
        if fn_load_weights is not None:
            fn_load_weights(model_dir)
        fn_run_training_episodes()
    
        fn_save_weights(model_dir)
        app_info.ENV.fn_close()
        return agent_mgr
        
    agent_mgr = namedtuple('_', [
        'fn_change_args',
        'fn_train',
        'APP_INFO',
    ])
    agent_mgr.fn_change_args = fn_change_args
    agent_mgr.fn_train = fn_train
    agent_mgr.APP_INFO = app_info

    return agent_mgr





