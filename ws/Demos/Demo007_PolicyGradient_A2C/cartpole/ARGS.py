from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
  {
      "STRATEGY": "D_PolicyGradientBased.ActorCritic.A2C",

      "NUM_EPOCHS": 1,
      "NUM_EPISODES": 12000,

      "MAX_STEPS_PER_EPISODE": 500,

      "GAMMA": 0.99,
      "LEARNING_RATE": 0.02,
      "EPSILON": 0.1,
      "CLIPPING_LOSS_RATIO": 0.2,
      "MAX_GRADIENT_NORM": 0.5,

      "UPDATE_STEP_INTERVAL" : 2000,

      "ENV_NAME": "CartPole-v1",

      "LOG_SKIP_INTERVAL": 10,
      "REWARD_GOAL": 500,
      "LOG_MEAN_INTERVAL": 1,
      "MAX_RESULT_COUNT": 100,

      "NUM_EPISODES_FOR_UPDATE": 1,
      "CONSECUTIVE_GOAL_HITS": 3,
      "ENV_SEED": 123,

      "AUTO_ARCHIVE": 1,
      "AUTO_INTERRUPT_HANDLING": 1,
  }
)

def fn_get_args():
    return app_info