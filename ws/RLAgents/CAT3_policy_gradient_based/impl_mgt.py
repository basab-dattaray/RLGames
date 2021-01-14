import os
import torch

from .Critic import Critic
from .misc import _fn_calculate_montecarlo_normalized_rewards

from ws.RLAgents.CAT3_policy_gradient_based.Buffer import Buffer
from ws.RLUtils.common.module_loader import load_function
from .model_mgt import model_mgt


def impl_mgt(app_info):
    # MODEL_ACTOR_NAME = 'Model_Actor.pth'
    # MODEL_CRITIC_NAME = 'Model_Critic.pth'

    _gamma = app_info['GAMMA']

    target_pkgpath = app_info.AGENT_FOLDER_PATH

    detail_mgt = load_function('detail_mgt', 'detail_mgt', target_pkgpath)
    fn_actor_loss_eval, fn_pick_action, fn_evaluate = detail_mgt(app_info)

    device = app_info.GPU_DEVICE

    target_pkgpath = app_info.AGENT_FOLDER_PATH
    Actor = load_function('Actor', 'Actor', target_pkgpath)
    _model_actor = Actor(app_info).to(device)
    _model_critic = Critic(app_info).to(device)
    #
    _optimizer = torch.optim.Adam(_model_actor.parameters(), lr=app_info.LEARNING_RATE, betas=(0.9, 0.999))
    #
    _model_old_critic = Critic(app_info).to(device)
    _model_old_actor = Actor(app_info).to(device)
    _buffer = Buffer()
    _update_interval_count = 0

    fn_save_to_neural_net, fn_load_from_neural_net = model_mgt(_model_actor, _model_critic)

    def fn_act(state):
        action = fn_pick_action(state, _buffer, _model_old_actor)
        return action

    def fn_add_transition(reward, done):
        _buffer.rewards.append(reward)
        _buffer.done.append(done)

    # def fn_load_from_neural_net(model_folder_path):
    #     nonlocal model_actor, model_critic
    #
    #     try:
    #         actor_dict = torch.load(os.path.join(model_folder_path, MODEL_ACTOR_NAME))
    #         model_actor.load_state_dict(actor_dict)
    #         critic_dict = torch.load(os.path.join(model_folder_path, MODEL_CRITIC_NAME))
    #         model_critic.load_state_dict(critic_dict)
    #         return True
    #     except Exception as x:
    #         return False
    #
    # def fn_save_to_neural_net(model_folder_path):
    #     nonlocal model_actor, model_critic
    #     if os.path.exists(model_folder_path) is False:
    #         os.makedirs(model_folder_path)
    #
    #     actor_path = os.path.join(model_folder_path, MODEL_ACTOR_NAME)
    #     torch.save(model_actor.state_dict(), actor_path)
    #
    #     critic_path = os.path.join(model_folder_path, MODEL_CRITIC_NAME)
    #     torch.save(model_critic.state_dict(), critic_path)

    def fn_should_update_network(done):
        nonlocal _update_interval_count
        _update_interval_count += 1
        if _update_interval_count % app_info['UPDATE_STEP_INTERVAL'] == 0:
            fn_update()

    def fn_update():
        nonlocal _model_old_actor, _model_old_critic

        # Monte Carlo rewards estimate:
        rewards = _fn_calculate_montecarlo_normalized_rewards(app_info, _buffer, _gamma)

        # make into tensors
        old_states = torch.stack(_buffer.states).to(device).detach()
        old_actions = torch.stack(_buffer.actions).to(device).detach()
        old_logprobs = torch.stack(_buffer.logprobs).to(device).detach()

        for _ in range(app_info['NUM_EPOCHS']):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = fn_evaluate(_model_actor, _model_critic, old_states, old_actions)
            loss = fn_actor_loss_eval(app_info, logprobs, old_logprobs, rewards, state_values)

            # calculate gradients and adjust weights
            _optimizer.zero_grad()
            loss.mean().backward()
            _optimizer.step()

        # update old policy model weights from current policy model weights
        _model_old_actor.load_state_dict(_model_actor.state_dict())
        _model_old_critic.load_state_dict(_model_critic.state_dict())

        _buffer.clear_buffer()

    return fn_act, fn_add_transition, fn_save_to_neural_net, fn_load_from_neural_net, fn_should_update_network
