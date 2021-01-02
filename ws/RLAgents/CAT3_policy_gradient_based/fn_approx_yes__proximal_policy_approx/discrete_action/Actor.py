import torch
from torch import nn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from ws.RLInterfaces.PARAM_KEY_NAMES import STATE_DIMENSIONS, ACTION_DIMENSIONS, ACTOR_HIDDEN_LAYER_NODES
import torch.nn.functional as F

from ws.RLUtils.modelling.hidden_layer_model_mgt import hidden_layer_model_mgt


class Actor(nn.Module):
    def __init__(self, app_info):
        super(Actor, self).__init__()

        self._hidden_layer_dims = app_info['ACTOR_HIDDEN_LAYER_NODES']
        fn_hidden_layers_input_proc, self.fn_hidden_layers_forward_proc = hidden_layer_model_mgt(self)

        env = app_info['ENV']
        action_size = env.fn_get_action_size() # app_info[ACTION_DIMENSIONS][0]
        state_size = env.fn_get_state_size() # app_info[STATE_DIMENSIONS][0]

        last_hidden_layer_size = fn_hidden_layers_input_proc(state_size)
        self.last_hidden_to_out = nn.Linear(last_hidden_layer_size, action_size)

    def forward(self, state):
        last_hidden_layer_data = torch.relu(self.fn_hidden_layers_forward_proc(state))
        z_data = self.last_hidden_to_out(last_hidden_layer_data)
        out_data = F.softmin(z_data, dim=-1)
        return out_data
