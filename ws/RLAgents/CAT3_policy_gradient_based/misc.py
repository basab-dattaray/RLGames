import torch

# from ws.RLInterfaces.PARAM_KEY_NAMES import *


def _fn_calculate_montecarlo_normalized_rewards(app_info, buffer, gamma):
    rewards = []
    discounted_reward = 0
    for reward, done in zip(reversed(buffer.rewards), reversed(buffer.done)):
        if done:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        rewards.insert(0, discounted_reward)
    # Normalizing the rewards:
    rewards = torch.tensor(rewards).to(app_info.GPU_DEVICE)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    return rewards


