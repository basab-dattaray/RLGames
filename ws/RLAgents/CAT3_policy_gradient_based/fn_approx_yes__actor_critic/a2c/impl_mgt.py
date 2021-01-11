import os

import torch

from ws.RLAgents.CAT3_policy_gradient_based.misc import _fn_calculate_montecarlo_normalized_rewards
from ws.RLAgents.CAT3_policy_gradient_based.Buffer import Buffer
from .ActorCritic import ActorCritic
from .detail_mgt import detail_mgt

def impl_mgt(app_info):
    MODEL_NAME = 'Model_ActorCritic.pth'

    _gamma = app_info['GAMMA']
    fn_actor_loss_eval, fn_pick_action, fn_evaluate = detail_mgt(app_info)

    _model_actor_critic = ActorCritic(app_info).to(app_info.GPU_DEVICE)

    _optimizer = torch.optim.Adam(_model_actor_critic.parameters(), lr=app_info.LEARNING_RATE, betas=(0.9, 0.999))

    _buffer = Buffer()
    _update_interval_count = 0

    def fn_act(state):
        action = fn_pick_action(state, _buffer, _model_actor_critic)
        return action

    def fn_add_transition(reward, done):
        _buffer.rewards.append(reward)
        _buffer.done.append(done)

    def fn_load_from_neural_net(current_folder_path):
        nonlocal _model_actor_critic
        if not os.path.exists(current_folder_path):
            return False

        model_name_path = os.path.join(current_folder_path, MODEL_NAME)

        if not os.path.exists(model_name_path):
            return False

        actor_dict = torch.load(model_name_path)


        _model_actor_critic.load_state_dict(actor_dict)
        return True


    def fn_save_to_neural_net(current_folder_path):
        nonlocal _model_actor_critic

        if os.path.exists(current_folder_path) is False:
            os.makedirs(current_folder_path)
        actor_critic_path = os.path.join(current_folder_path, MODEL_NAME)
        torch.save(_model_actor_critic.state_dict(), actor_critic_path)

    def fn_should_update_network(done):
        nonlocal _update_interval_count
        _update_interval_count += 1
        if _update_interval_count % app_info['UPDATE_STEP_INTERVAL'] == 0:
            fn_update()

    def fn_update():
        nonlocal _model_actor_critic

        # Monte Carlo rewards estimate:
        rewards = _fn_calculate_montecarlo_normalized_rewards(app_info, _buffer, _gamma)

        for _ in range(app_info['NUM_EPOCHS']):
            # Evaluating old actions and values :
            logprobs, state_values = fn_evaluate(_buffer)
            loss = fn_actor_loss_eval(logprobs, rewards, state_values)

            # calculate gradients and adjust weights
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

        _buffer.clear_buffer()

    return fn_act, fn_add_transition, fn_save_to_neural_net, fn_load_from_neural_net, fn_should_update_network
