from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
  {
    "STRATEGY": "B_ValueBased.Bootstrapping.OnPolicy.sarsa",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_sarsa",
  }
)

def fn_get_args():
    return app_info