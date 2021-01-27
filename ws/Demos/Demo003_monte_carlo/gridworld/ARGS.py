from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
  {
    "STRATEGY": "CAT2_policy_based.fn_approx_no__on_policy__sampling.monte_carlo",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_monte_carlo",

  }
)

def fn_get_args():
    return app_info