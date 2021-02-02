from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
  {
    "STRATEGY": "C_ValueBase_WithFunctionApproximation.OnPolicy_Bootstrapping.sarsa",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_sarsa",
  }
)

def fn_get_args():
    return app_info