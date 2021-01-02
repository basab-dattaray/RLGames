import torch.nn as nn
import torch.nn.functional as F

# from ws.wsRLInterfaces.PARAM_KEY_NAMES import STATE_DIMENSIONS, ACTION_DIMENSIONS


class ActorCritic(nn.Module):
    def __init__(self, app_info):

        super(ActorCritic, self).__init__()
        env = app_info['ENV']
        action_size = env.fn_get_action_size() # app_info[ACTION_DIMENSIONS][0]
        state_size = env.fn_get_state_size() # app_info[STATE_DIMENSIONS][0]
        hidden_layer_size = 256

        self._app_info = app_info

        self.state_to_hidden = nn.Linear(state_size, hidden_layer_size)
        self.action_layer = nn.Linear(hidden_layer_size, action_size)
        self.value_layer = nn.Linear(hidden_layer_size, 1)
        self.state_value = None
        # self.state_values = []

    def forward(self, state):
        action_info = None

        hidden = self.state_to_hidden(state)
        state = F.relu(hidden)

        self.state_value = self.value_layer(state)

        action_info = self.action_layer(state)
        policy = F.softmax(action_info)
        return policy

        return None

    def get_state_value(self):
        return self.state_value
