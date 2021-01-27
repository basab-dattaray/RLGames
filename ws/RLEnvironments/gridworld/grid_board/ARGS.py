from ws.RLUtils.common.DotDict import DotDict
app_info = DotDict({
    "DISPLAY": {
        "APP_NAME": "Policy Iterator",
        "BOARD_BLOCKERS": [{"x": 1, "y": 2, "reward": -100}, {"x": 2, "y": 1, "reward": -100}],
        "BOARD_GOAL": {"x": 2, "y": 2, "reward": 100},
        "UNIT": 100,
        "WIDTH": 6,
        "HEIGHT": 5
    },
})