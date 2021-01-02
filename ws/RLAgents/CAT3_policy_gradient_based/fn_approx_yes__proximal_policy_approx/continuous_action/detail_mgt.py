import torch

# from ws.RLInterfaces.PARAM_KEY_NAMES import CLIPPING_LOSS_RATIO, ACTION_DIMENSIONS, GPU_DEVICE
from torch.distributions import MultivariateNormal


def detail_mgt(app_info):
    device = app_info['GPU_DEVICE']

    def fn_actor_loss_eval(app_info, logprobs, old_logprobs, rewards, state_values):
        clipping_loss_ratio = app_info['CLIPPING_LOSS_RATIO']
        # same as (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - old_logprobs.detach())
        # surrogate losses:
        advantages = rewards - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clipping_loss_ratio, 1 + clipping_loss_ratio) * advantages
        loss = -torch.min(surr1, surr2)  # + 0.5.2*_MseLoss(state_values, rewards) - 0.01*dist_entropy
        return loss

    def fn_pick_action(state, buffer, policy_old_actor):
        tensored_state = torch.from_numpy(state).float().to(device)
        action_mean = policy_old_actor.forward(tensored_state)
        action_size = app_info['ACTION_DIMENSIONS']

        # action_std = app_info[SD_FOR_MULTIVARIATE_NORMAL_ACTION_DISTRIBUTION]
        action_var = policy_old_actor.get_action_var()
        cov_mat = torch.diag(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        buffer.states.append(tensored_state)
        buffer.actions.append(action)
        buffer.logprobs.append(action_logprob)

        result = action.detach().cpu().numpy()

        return result

    def fn_evaluate(policy_actor, policy_critic, state, action):
        action_mean = torch.squeeze(policy_actor.forward(state))

        action_var = policy_actor.get_action_var().expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        #### same below here
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()

        state_value = policy_critic.forward(state)
        # state_value = policy_critic.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    return fn_actor_loss_eval, fn_pick_action, fn_evaluate
