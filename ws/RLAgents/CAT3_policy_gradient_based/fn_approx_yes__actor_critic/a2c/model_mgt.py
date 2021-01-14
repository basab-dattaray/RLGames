import os

import torch


def model_mgt(_model_actor_critic):
    MODEL_NAME = 'Model_ActorCritic.pth'
    def fn_load_from_neural_net(model_folder_path):

        if not os.path.exists(model_folder_path):
            return False

        model_name_path = os.path.join(model_folder_path, MODEL_NAME)

        if not os.path.exists(model_name_path):
            return False

        actor_dict = torch.load(model_name_path)


        _model_actor_critic.load_state_dict(actor_dict)
        return True


    def fn_save_to_neural_net(model_folder_path):

        if os.path.exists(model_folder_path) is False:
            os.makedirs(model_folder_path)
        actor_critic_path = os.path.join(model_folder_path, MODEL_NAME)
        torch.save(_model_actor_critic.state_dict(), actor_critic_path)

    return fn_save_to_neural_net, fn_load_from_neural_net