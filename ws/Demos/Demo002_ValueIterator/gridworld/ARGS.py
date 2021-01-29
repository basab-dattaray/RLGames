from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict({
    "STRATEGY": "Category1_ModelBased.PlanningBased.value_iterator",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_iteration_based",

})

def fn_get_args():
    return app_info