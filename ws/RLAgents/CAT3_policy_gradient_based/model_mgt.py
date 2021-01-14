import os
from collections import namedtuple

import torch

from ws.RLUtils.common.module_loader import load_function
from .Critic import Critic

MODEL_ACTOR_NAME = 'Model_Actor.pth'
MODEL_CRITIC_NAME = 'Model_Critic.pth'
def model_mgt(app_info):

    device = app_info.GPU_DEVICE

    target_pkgpath = app_info.AGENT_FOLDER_PATH
    Actor = load_function('Actor', 'Actor', target_pkgpath)
    _model_actor = Actor(app_info).to(device)
    _model_critic = Critic(app_info).to(device)

    def fn_load_from_neural_net(model_folder_path):
        nonlocal _model_actor, _model_critic

        try:
            actor_dict = torch.load(os.path.join(model_folder_path, MODEL_ACTOR_NAME))
            _model_actor.load_state_dict(actor_dict)
            critic_dict = torch.load(os.path.join(model_folder_path, MODEL_CRITIC_NAME))
            _model_critic.load_state_dict(critic_dict)
            return True
        except Exception as x:
            return False

    def fn_save_to_neural_net(model_folder_path):
        nonlocal _model_actor, _model_critic
        if os.path.exists(model_folder_path) is False:
            os.makedirs(model_folder_path)

        actor_path = os.path.join(model_folder_path, MODEL_ACTOR_NAME)
        torch.save(_model_actor.state_dict(), actor_path)

        critic_path = os.path.join(model_folder_path, MODEL_CRITIC_NAME)
        torch.save(_model_critic.state_dict(), critic_path)

    model_mgr = namedtuple('_', ['fn_save_to_neural_net', 'fn_load_from_neural_net'])
    model_mgr.fn_save_to_neural_net = fn_save_to_neural_net
    model_mgr.fn_load_from_neural_net = fn_load_from_neural_net

    return model_mgr