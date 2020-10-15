import torch

from ws.RLInterfaces.PARAM_KEY_NAMES import CLIPPING_LOSS_RATIO, GPU_DEVICE
from torch.distributions import Categorical


def detail_mgt(app_info):
    def fn_actor_loss_eval(app_info, logprobs, old_logprobs, rewards, state_values):
        clipping_loss_ratio = app_info[CLIPPING_LOSS_RATIO]
        # same as (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - old_logprobs.detach())
        # surrogate losses:
        advantages = rewards - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clipping_loss_ratio, 1 + clipping_loss_ratio) * advantages
        loss = -torch.min(surr1, surr2)  # + 0.5.2*_MseLoss(state_values, rewards) - 0.01*dist_entropy
        return loss

    def fn_pick_action(state, buffer, policy_old_actor):
        state = torch.from_numpy(state).float().to(app_info[GPU_DEVICE])
        action_probs = policy_old_actor.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample()

        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.logprobs.append(dist.log_prob(action))

        return action.item()

    def fn_evaluate(policy_actor, policy_critic, states, actions):
        action_probs = policy_actor.forward(states)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        state_value = policy_critic.forward(states)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    return fn_actor_loss_eval, fn_pick_action, fn_evaluate
