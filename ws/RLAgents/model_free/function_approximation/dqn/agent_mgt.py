from ws.RLInterfaces.PARAM_KEY_NAMES import NUM_EPISODES, MAX_STEPS_PER_EPISODE

import numpy as np

from ws.RLAgents.model_free.function_approximation.dqn.impl_mgt import impl_mgr


def agent_mgr(app_info, env):
    _action_size = env.fnGetActionDimensions()[0]
    _state_size = env.fnGetStateDimensions()[0]

    fnReset, fn_remember, fnAct, fnReplay, fnSaveWeights, fnLoadWeights = impl_mgr(app_info, _state_size, _action_size)

    def fnTrain():
        # nonlocal fn_remember

        num_episodes = app_info[NUM_EPISODES]
        loss = []
        for e in range(num_episodes):
            state = env.fnReset()
            state = np.reshape(state, (1, _state_size))
            score = 0

            num_steps = 0
            for i in range(app_info[MAX_STEPS_PER_EPISODE]):
                num_steps += 1
                action = fnAct(state)
                env.fnDoRender()
                next_state, reward, done, _ = env.fnStep(action)
                score += reward
                next_state = np.reshape(next_state, (1, _state_size))
                fn_remember(state, action, reward, next_state, done)
                state = next_state
                fnReplay()
                if done or (i == app_info[MAX_STEPS_PER_EPISODE] - 1):
                    print("episode: {}/{}, num of steps: {}, score: {}".format(e, num_episodes, num_steps, score))
                    break
            loss.append(score)

        return loss

    return fnTrain, fnSaveWeights, fnLoadWeights



