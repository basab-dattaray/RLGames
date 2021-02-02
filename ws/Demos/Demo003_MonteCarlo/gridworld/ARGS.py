from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
  {
    "STRATEGY": "C_ValueBase_WithFunctionApproximation.OnPolicy_Sampling.monte_carlo",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_monte_carlo",

  }
)

def fn_get_args():
    return app_info