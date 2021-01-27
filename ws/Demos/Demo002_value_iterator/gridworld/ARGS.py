from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict({
    "STRATEGY": "CAT1_model_based.planning.value_iterator",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_iteration_based",

})

def fn_get_args():
    return app_info