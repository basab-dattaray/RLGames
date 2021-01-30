import os
from collections import namedtuple

import numpy as np

# from ws.RLUtils.setup.preparation_mgt import preparation_mgt
from ws.RLUtils.setup.startup_mgt import startup_mgt
from .impl_mgt import impl_mgt

def agent_mgt(caller_file):
    # app_info, env = preparation_mgt(caller_file)
    app_info = startup_mgt(caller_file, __file__)
    env = app_info.ENV
    _action_size = env.fn_get_action_size()
    _state_size = env.fn_get_state_size()

    fn_reset_env, fn_remember, fnAct, fnReplay, fnSaveWeights, fnLoadWeights = impl_mgt(app_info, _state_size, _action_size)
    def fn_train():
        def fn_run_training_episodes():
            # nonlocal fn_remember
    
            num_episodes = app_info.NUM_EPISODES
            loss = []
            for episode_num in range(1, num_episodes):
                state = env.fn_reset_env()
                state = np.reshape(state, (1, _state_size))
                score = 0

                for step_num in range(app_info.MAX_STEPS_PER_EPISODE):
                    action = fnAct(state)
                    env.fn_render()
                    next_state, reward, done, _ = env.fn_take_step(action)
                    score += reward
                    next_state = np.reshape(next_state, (1, _state_size))
                    fn_remember(state, action, reward, next_state, done)
                    state = next_state
                    fnReplay()
    
                    if done or (step_num == app_info.MAX_STEPS_PER_EPISODE - 1) or (score >= app_info.REWARD_GOAL):
                        print("episode: {}/{}, num of steps: {}, score: {}".format(episode_num, num_episodes, step_num, score))
                        break
                loss.append(score)
    
            return loss
    
        cwd = __file__.rsplit('/', 1)[0]
        model_dir = os.path.join(cwd, "models")
        fnLoadWeights(model_dir)
        fn_run_training_episodes()
    
        fnSaveWeights(model_dir)
        env.fn_close()
        return agent_mgr
        
    agent_mgr = namedtuple('_', [
        'fn_train',
    ])

    agent_mgr.fn_train = fn_train

    return agent_mgr





