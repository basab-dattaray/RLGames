import gym

# from ws.RLInterfaces.PARAM_KEY_NAMES import ENV_NAME


class env_mgt:
    def __init__(self, app_info):
        self._env = gym.make(app_info['ENV_NAME'])
        self._state_size = self._env.observation_space.shape[0]
        self._action_size = self._env.action_space.n

    def fn_reset_env(self):
        return self._env.reset()

    def fn_take_step(self, action):
        # nonlocal _state
        next_state, reward, done, _ = self._env.step(action)
        return next_state, reward, done, None

    def fn_render(self):
        self._env.render()

    def fn_get_state_size(self):
        return [self._state_size][0]

    def fn_get_action_size(self):
        return [self._action_size][0]

    # def fnInnerEnv(self):
    #     return self._env

    def fn_close(self):
        self._env.close()

    # def fnGetDisplayController():
    #     return None

    # return {
    #     envMgr__fnReset: fn_reset_env,
    #     envMgr__fnStep: fn_take_step,
    #     envMgr__fnDisplay: fn_render,
    #     envMgr__fnGetStateDimensions: fn_get_state_size,
    #     envMgr__fnGetActionDimensions: fn_get_action_size,
    #     envMgr__fnInnerEnv: fnInnerEnv,
    #     envMgr__fnClose: fn_close,
    #     envMgr__fnGetDisplayController: fnGetDisplayController
    # }
