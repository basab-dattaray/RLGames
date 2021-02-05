import torch

from ws.RLAgents.D_PolicyGradientBased.ProximalPolicyOptimization.continuous_action.Critic import Critic
from ws.RLAgents.D_PolicyGradientBased.misc import _fn_calculate_montecarlo_normalized_rewards

from ws.RLAgents.D_PolicyGradientBased.Buffer import Buffer
from ws.RLUtils.common.module_loader import load_function, load_mgt_function
from ws.RLAgents.D_PolicyGradientBased.ProximalPolicyOptimization.model_mgt import model_mgt


def impl_mgt(app_info):

    _gamma = app_info.GAMMA

    detail_mgt = load_mgt_function(loc_dotpath= app_info.AGENT_DOT_PATH, module_name='detail_mgt')

    fn_actor_loss_eval, fn_pick_action, fn_evaluate = detail_mgt(app_info)

    device = app_info.GPU_DEVICE


    Actor = load_mgt_function(loc_dotpath= app_info.AGENT_DOT_PATH, module_name='Actor')
    _model_actor = Actor(app_info).to(device)
    _model_critic = Critic(app_info).to(device)
    #
    _optimizer = torch.optim.Adam(_model_actor.parameters(), lr=app_info.LEARNING_RATE, betas=(0.9, 0.999))
    #
    _model_old_critic = Critic(app_info).to(device)
    _model_old_actor = Actor(app_info).to(device)
    _buffer = Buffer()
    _update_interval_count = 0

    fn_save_model, fn_load_model = model_mgt(app_info.RESULTS_PATH_, _model_actor, _model_critic)

    def fn_act(state):
        action = fn_pick_action(state, _buffer, _model_old_actor)
        return action

    def fn_add_transition(reward, done):
        _buffer.rewards.append(reward)
        _buffer.done.append(done)

    def fn_should_update_network(done):
        nonlocal _update_interval_count
        _update_interval_count += 1
        if _update_interval_count % app_info.UPDATE_STEP_INTERVAL == 0:
            fn_update()

    def fn_update():
        nonlocal _model_old_actor, _model_old_critic

        # Monte Carlo rewards estimate:
        rewards = _fn_calculate_montecarlo_normalized_rewards(app_info, _buffer, _gamma)

        # make into tensors
        old_states = torch.stack(_buffer.states).to(device).detach()
        old_actions = torch.stack(_buffer.actions).to(device).detach()
        old_logprobs = torch.stack(_buffer.logprobs).to(device).detach()

        for _ in range(app_info.NUM_EPOCHS):
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

    return fn_act, fn_add_transition, fn_save_model, fn_load_model, fn_should_update_network
