from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict({
    "STRATEGY": "A_ModelBased.Planning.policy_iterator",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_iteration_based",

    "AUTO_ARCHIVE": 1,
    "AUTO_INTERRUPT_HANDLING": 1,

})

def fn_get_args():
    return app_info