import gym

# from ws.RLInterfaces.PARAM_KEY_NAMES import ENV_NAME


class Env:
    def __init__(self, app_info):
        self._env = gym.make(app_info['ENV_NAME'])
        self._state_size = self._env.observation_space.shape[0]
        self._action_size = self._env.action_space.n

    def fnReset(self):
        return self._env.reset()

    def fnStep(self, action):
        # nonlocal _state
        next_state, reward, done, _ = self._env.step(action)
        return next_state, reward, done, None

    def fnDoRender(self):
        self._env.render()

    def fnGetStateDimensions(self):
        return [self._state_size][0]

    def fnGetActionDimensions(self):
        return [self._action_size][0]

    def fnInnerEnv(self):
        return self._env

    def fnClose(self):
        self._env.close()

    # def fnGetDisplayController():
    #     return None

    # return {
    #     envMgr__fnReset: fnReset,
    #     envMgr__fnStep: fnStep,
    #     envMgr__fnDisplay: fnDoRender,
    #     envMgr__fnGetStateDimensions: fnGetStateDimensions,
    #     envMgr__fnGetActionDimensions: fnGetActionDimensions,
    #     envMgr__fnInnerEnv: fnInnerEnv,
    #     envMgr__fnClose: fnClose,
    #     envMgr__fnGetDisplayController: fnGetDisplayController
    # }
