import os
from collections import namedtuple

import torch

MODEL_ACTOR_NAME = 'Model_Actor.pth'
MODEL_CRITIC_NAME = 'Model_Critic.pth'
def model_mgt(model_actor, model_critic):

    def fn_load_from_neural_net(model_folder_path= None, model_file_name= None):
        # nonlocal model_actor, model_critic

        try:
            actor_dict = torch.load(os.path.join(model_folder_path, MODEL_ACTOR_NAME))
            model_actor.load_state_dict(actor_dict)
            critic_dict = torch.load(os.path.join(model_folder_path, MODEL_CRITIC_NAME))
            model_critic.load_state_dict(critic_dict)
            return True
        except Exception as x:
            return False

    def fn_save_to_neural_net(model_folder_path= None, model_file_name= None):
        # nonlocal model_actor, model_critic
        if os.path.exists(model_folder_path) is False:
            os.makedirs(model_folder_path)

        actor_path = os.path.join(model_folder_path, MODEL_ACTOR_NAME)
        torch.save(model_actor.state_dict(), actor_path)

        critic_path = os.path.join(model_folder_path, MODEL_CRITIC_NAME)
        torch.save(model_critic.state_dict(), critic_path)

    return fn_save_to_neural_net, fn_load_from_neural_net