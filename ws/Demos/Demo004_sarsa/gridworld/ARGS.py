from ws.RLUtils.common.DotDict import DotDict

args = DotDict(
{
  "STRATEGY": "CAT2_value_based.fn_approx_no__on_policy__bootstrap.sarsa",
  "ENV_NAME": "Gridworld-v1",
  "AGENT_CONFIG": "gridwell_1",

  "NUM_EPISODES": 500,
  "BATCH_SIZE": 64,
  "GAMMA": 0.99,
  "LEARNING_RATE": 0.01,
  "EPSILON": 0.1,
  "RHO": 0.99,
  "DISCOUNT_FACTOR": 0.9,

  "ENV_DISPLAY_ON": 1,

  "display": {
    "APP_NAME": "SARSA",
    "BOARD_BLOCKERS": [  {"x": 1, "y": 2, "reward": -100},  {"x": 2, "y": 1, "reward": -100}],
    "BOARD_GOAL": {"x":2,"y":2, "reward": 100},
    "UNIT": 100,
    "WIDTH": 6,
    "HEIGHT": 5
  }
}
)

def fn_get_args():
    return args