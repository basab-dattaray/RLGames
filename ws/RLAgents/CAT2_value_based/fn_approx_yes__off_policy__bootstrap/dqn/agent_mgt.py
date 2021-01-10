import os

import numpy as np

from ws.RLUtils.setup.preparation_mgt import preparation_mgt
from .impl_mgt import impl_mgt

def agent_mgt(caller_file):
    app_info, env = preparation_mgt(caller_file)
    _action_size = env.fn_get_action_size()
    _state_size = env.fn_get_state_size()

    fn_reset_env, fn_remember, fnAct, fnReplay, fnSaveWeights, fnLoadWeights = impl_mgt(app_info, _state_size, _action_size)

    def fnTrain():
        # nonlocal fn_remember

        # num_episodes = app_info['NUM_EPISODES']
        loss = []
        for episode_num in range(1, app_info['NUM_EPISODES']):
            state = env.fn_reset_env()
            state = np.reshape(state, (1, _state_size))
            score = 0
            # print(f'EPISODE NUM: {episode_num}')
            num_steps = 1
            MAX_NUM_EPISODES = app_info['MAX_STEPS_PER_EPISODE'] + 1
            reward = -9999
            while (num_steps <= app_info['MAX_STEPS_PER_EPISODE']) or (reward >= app_info['REWARD_GOAL']):

                action = fnAct(state)
                if episode_num ==  app_info['NUM_EPISODES']:
                    env.fn_render()

                next_state, reward, done, _ = env.fn_take_step(action)
                score += reward
                next_state = np.reshape(next_state, (1, _state_size))
                fn_remember(state, action, reward, next_state, done)
                state = next_state
                fnReplay()
                num_steps += 1

            print("episode: {}/{}, num of steps: {}, score: {}".format(episode_num, app_info['NUM_EPISODES'], num_steps, score))
            loss.append(score)

        return loss

    cwd = __file__.rsplit('/', 1)[0]
    model_dir = os.path.join(cwd, "models")
    fnLoadWeights(model_dir)
    fnTrain()

    fnSaveWeights(model_dir)
    env.fn_close()

    # return fnTrain, fnSaveWeights, fnLoadWeights



