from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
  {
    "STRATEGY": "B_ValueBased.Sampling.OnPolicy.monte_carlo",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_monte_carlo",

  }
)

def fn_get_args():
    return app_info