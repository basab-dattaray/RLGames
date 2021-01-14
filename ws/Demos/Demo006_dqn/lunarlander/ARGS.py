from ws.RLUtils.common.DotDict import DotDict

args = DotDict(
  {
    "STRATEGY": "CAT2_value_based.fn_approx_yes__off_policy__bootstrap.dqn",

    "NUM_EPISODES": 20,
    "BATCH_SIZE": 64,
    "GAMMA": 0.99,
    "LEARNING_RATE": 0.001,
    "EPSILON": 1.0,
    "RHO": 0.99,

    "ENV_NAME": "LunarLander-v2",
    "ENV_DISPLAY_ON": 1,

    "REWARD_GOAL": 210,

    "MAX_STEPS_PER_EPISODE": 5000,

    "DEQUE_MEM_SIZE": 4000,

    "EPSILON_MIN": 0.01,
    "EPSILON_DECAY": 0.996

  }
)

def fn_get_args():
    return args