# from ws.RLInterfaces.PARAM_KEY_NAMES import NUM_EPISODES, MAX_STEPS_PER_EPISODE

import numpy as np

# from ws.RLAgents.model_free.function_approximation.dqn.impl_mgt import impl_mgt
from ws.RLAgents.CAT2_value_based.fn_approx_yes__off_policy__bootstrap.dqn.impl_mgt import impl_mgt


def agent_mgt(app_info, env):
    _action_size = env.fn_get_action_size()
    _state_size = env.fn_get_state_size()

    fn_reset_env, fn_remember, fnAct, fnReplay, fnSaveWeights, fnLoadWeights = impl_mgt(app_info, _state_size, _action_size)

    def fnTrain():
        # nonlocal fn_remember

        num_episodes = app_info['NUM_EPISODES']
        loss = []
        for e in range(num_episodes):
            state = env.fn_reset_env()
            state = np.reshape(state, (1, _state_size))
            score = 0

            num_steps = 0
            for i in range(app_info['MAX_STEPS_PER_EPISODE']):
                num_steps += 1
                action = fnAct(state)
                env.fn_render()
                next_state, reward, done, _ = env.fn_take_step(action)
                score += reward
                next_state = np.reshape(next_state, (1, _state_size))
                fn_remember(state, action, reward, next_state, done)
                state = next_state
                fnReplay()
                if done or (i == app_info['MAX_STEPS_PER_EPISODE'] - 1):
                    print("episode: {}/{}, num of steps: {}, score: {}".format(e, num_episodes, num_steps, score))
                    break
            loss.append(score)

        return loss

    return fnTrain, fnSaveWeights, fnLoadWeights



