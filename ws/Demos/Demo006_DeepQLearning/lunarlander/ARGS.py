from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
  {
    "STRATEGY": "Category2_ValueBased.DeepLearning_OffPolicy_Bootstrapping.dqn",

    "NUM_EPISODES": 20,
    "BATCH_SIZE": 64,
    "GAMMA": 0.99,
    "LEARNING_RATE": 0.001,
    "EPSILON": 1.0,
    "RHO": 0.99,

    "ENV_NAME": "LunarLander-v2",

    "REWARD_GOAL": 210,

    "MAX_STEPS_PER_EPISODE": 5000,

    "DEQUE_MEM_SIZE": 4000,

    "EPSILON_MIN": 0.01,
    "EPSILON_DECAY": 0.996,

    "FORCE_CPU_USE": 1,

    "AUTO_ARCHIVE": 1,
    "AUTO_INTERRUPT_HANDLING": 1,


  }
)

def fn_get_args():
    return app_info