from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict({
    "STRATEGY": "CAT1_model_based.planning.policy_iterator",
    "ENV_NAME": "Gridworld-v1",
    "AGENT_CONFIG": "gridwell_1",

    "DISCOUNT_FACTOR": 0.9,

    "DISPLAY": {
        "APP_NAME": "Policy Iterator",
        "BOARD_BLOCKERS": [{"x": 1, "y": 2, "reward": -100}, {"x": 2, "y": 1, "reward": -100}],
        "BOARD_GOAL": {"x": 2, "y": 2, "reward": 100},
        "UNIT": 100,
        "WIDTH": 6,
        "HEIGHT": 5
    },
    "AUTO_ARCHIVE": 1,
    "AUTO_INTERRUPT_HANDLING": 1,
})

def fn_get_args():
    return app_info