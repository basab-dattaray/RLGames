import torch
from torch import nn

# from ws.RLInterfaces.PARAM_KEY_NAMES_NAMES import STATE_DIMENSIONS, CRITIC_HIDDEN_LAYER_NODES

from ws.RLUtils.modelling.hidden_layer_model_mgt import hidden_layer_model_mgt


class Critic(nn.Module):
    def __init__(self, app_info):
        super(Critic, self).__init__()

        self._hidden_layer_dims = app_info.CRITIC_HIDDEN_LAYER_NODES
        fn_hidden_layers_input_proc, self.fn_hidden_layers_forward_proc = hidden_layer_model_mgt(self)

        env = app_info.ENV
        state_size = env.fn_get_state_size()

        last_hidden_layer_size = fn_hidden_layers_input_proc(state_size)

        self.last_hidden_to_out = nn.Linear(last_hidden_layer_size, 1)

    def forward(self, state):
        last_hidden_layer_data = self.fn_hidden_layers_forward_proc(state)
        out = torch.tanh(self.last_hidden_to_out(last_hidden_layer_data))
        return out
