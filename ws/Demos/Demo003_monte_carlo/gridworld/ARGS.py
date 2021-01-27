from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
{
  "STRATEGY": "CAT2_policy_based.fn_approx_no__on_policy__sampling.monte_carlo",
  "ENV_NAME": "Gridworld-v1",
  "AGENT_CONFIG": "gridwell_monte_carlo",


  "DISPLAY": {
    "APP_NAME": "Monte Carlo",
    "BOARD_BLOCKERS": [  {"x": 1, "y": 2, "reward": -100},  {"x": 2, "y": 1, "reward": -100}],
    "BOARD_GOAL": {"x":2,"y":2, "reward": 100},
    "UNIT": 100,
    "WIDTH": 6,
    "HEIGHT": 5
  }
}
)

def fn_get_args():
    return app_info