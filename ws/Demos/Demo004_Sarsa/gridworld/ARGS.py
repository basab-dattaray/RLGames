from ws.RLUtils.common.app_info_lib import DotDict

app_info = DotDict(
  {
    "STRATEGY": "B_ValueBased.Bootstrapping.OnPolicy.sarsa",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_sarsa",

    "AUTO_ARCHIVE": 1,
    "AUTO_INTERRUPT_HANDLING": 1,
  }
)

def fn_get_args():
    return app_info