import torch

from ws.RLInterfaces.PARAM_KEY_NAMES import GPU_DEVICE
from torch.distributions import Categorical
import torch.nn.functional as F


def detail_mgr(app_info):
    def fn_actor_loss_eval(logprobs, rewards, state_values):
        advantage = rewards.view(-1, 1) - torch.stack(state_values)
        actor_loss = -(torch.stack(logprobs).view(-1, 1) * advantage)
        critic_loss = F.smooth_l1_loss(torch.stack(state_values), rewards).view(-1, 1)
        loss = actor_loss + critic_loss
        cummulative_loss = loss.sum()
        return cummulative_loss

    def fn_pick_action(state, buffer, model):
        state = torch.from_numpy(state).float().to(app_info[GPU_DEVICE])
        action_prob = model.forward(state)
        state_value = model.get_state_value()
        buffer.state_values.append(state_value)

        action_distribution = Categorical(action_prob)
        t_action = action_distribution.sample()

        action_log_prob = action_distribution.log_prob(t_action)

        action = int(t_action.item())
        buffer.actions.append(t_action)
        buffer.logprobs.append(action_log_prob)
        return action

    def fn_evaluate(buffer):
        return buffer.logprobs, buffer.state_values

    return fn_actor_loss_eval, fn_pick_action, fn_evaluate
