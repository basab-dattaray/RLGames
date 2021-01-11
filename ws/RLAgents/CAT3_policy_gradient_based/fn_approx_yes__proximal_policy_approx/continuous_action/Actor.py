import torch
from torch import nn

# from ws.RLInterfaces.PARAM_KEY_NAMES import STATE_DIMENSIONS, ACTION_DIMENSIONS, \
#     SD_FOR_MULTIVARIATE_NORMAL_ACTION_DISTRIBUTION, ACTOR_HIDDEN_LAYER_NODES, GPU_DEVICE
from ws.RLUtils.modelling.hidden_layer_model_mgt import hidden_layer_model_mgt

class Actor(nn.Module):
    def __init__(self, app_info):
        self._app_info = app_info
        super(Actor, self).__init__()

        self._hidden_layer_dims = app_info.ACTOR_HIDDEN_LAYER_NODES
        fnHiddenLayersInputProc, self.fn_hidden_layers_forward_proc = hidden_layer_model_mgt(self)

        self.action_size = app_info.ACTION_DIMENSIONS
        state_size = app_info.STATE_DIMENSIONS
        self.action_std = app_info.SD_FOR_MULTIVARIATE_NORMAL_ACTION_DISTRIBUTION

        last_hidden_layer_size = fnHiddenLayersInputProc(state_size)
        self.last_hidden_to_out = nn.Linear(last_hidden_layer_size, self.action_size)

    def forward(self, state):
        out = None

        last_hidden_layer_data = torch.tanh(self.fn_hidden_layers_forward_proc(state))
        out = torch.tanh(self.last_hidden_to_out(last_hidden_layer_data))

        self.action_var = torch.full((self.action_size,), self.action_std * self.action_std).to(self._app_info.GPU_DEVICE)
        return out

    def get_action_var(self):
        return self.action_var

