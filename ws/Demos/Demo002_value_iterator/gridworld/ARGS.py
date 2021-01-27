from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict({
    "STRATEGY": "CAT1_model_based.planning.value_iterator",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_iteration_based",

    "DISPLAY": {
        "APP_NAME": "Value Iterator",
        "BOARD_BLOCKERS": [{"x": 1, "y": 2, "reward": -100}, {"x": 2, "y": 1, "reward": -100}],
        "BOARD_GOAL": {"x": 2, "y": 2, "reward": 100},
        "UNIT": 100,
        "WIDTH": 6,
        "HEIGHT": 5
    }
})

def fn_get_args():
    return app_info