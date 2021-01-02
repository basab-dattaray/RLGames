from ws.RLEnvironments.gridworld.logic.SETUP_INFO import POSSIBLE_ACTIONS

from .logic.Episode import Episode

# from ws.RLInterfaces.PARAM_KEY_NAMES import OBJ_EPISODE


class Env:

    def __init__(self, app_info):
        self._app_info = app_info
        self._episode = None

        self._board_blockers = app_info["display"]["BOARD_BLOCKERS"]
        self._board_goal = app_info["display"]["BOARD_GOAL"]
        self._width = app_info["display"]["WIDTH"]
        self._height = app_info["display"]["HEIGHT"]

        self._reward = None
        self._possible_actions = None
        self._all_states = None
        self._state = None

        self.fnReset()

    def fnReset(self):
        self._episode = Episode()
        self._app_info['OBJ_EPISODE'] = self._episode

        self._reward = [[0] * self._width for _ in range(self._height)]
        self._possible_actions = POSSIBLE_ACTIONS

        for blocker in self._board_blockers:
            self._reward[blocker['y']][blocker['x']] = blocker['reward']  # for square

        self._reward[self._board_goal['y']][self._board_goal['x']] = self._board_goal['reward']  # for triangle
        self._all_states = []

        for x in range(self._width):
            for y in range(self._height):
                state = [x, y]
                self._all_states.append(state)

        self._state = [0, 0]
        return self._state

    def fnSetState(self, state):
        self._state = state

    def fnGetAllStates(self):
        return self._all_states

    def fn_value_table_possible_actions(self):
        return self._possible_actions

    def step(self, action):
        one = 1
        next_state_x = self._state[0]
        next_state_y = self._state[1]

        if action == 0:  # up
            if self._state[1] >= one:
                next_state_y -= one
        elif action == 1:  # down
            if self._state[1] < (self._height - 1) * one:
                next_state_y += one
        elif action == 2:  # left
            if self._state[0] >= one:
                next_state_x -= one
        elif action == 3:  # right
            if self._state[0] < (self._width - 1) * one:
                next_state_x += one

        return next_state_x, next_state_y

    def fnStep(self, action, planning_mode=False):
        next_state = self.step(action)
        reward = self._reward[next_state[1]][next_state[0]]

        self._episode.fn_update_episode(reward)

        if not planning_mode:
            self._state = next_state

        return next_state, reward, self._episode.fn_get_episode_status(), None

    # def fnDisplayDevice(self):
    #     return self._fn_display_device

    def fnDoRender(self):
        pass

    def fnGetStateDimensions(self):
        return [self._width, self._height]

    @staticmethod
    def fnGetActionDimensions():
        return [2]

    @staticmethod
    def fnInnerEnv():
        return None

    def fnClose(self):
        pass
