from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
    {
        "policy_iterator": {
            "DISPLAY": {
                "BOARD_BLOCKERS": [{"x": 1, "y": 2, "reward": -100}, {"x": 2, "y": 1, "reward": -100}],
                "BOARD_GOAL": {"x": 2, "y": 2, "reward": 100},
                "UNIT": 100,
                "WIDTH": 6,
                "HEIGHT": 5
            }
        },
        "value_iterator": {
            "DISPLAY": {
                "BOARD_BLOCKERS": [{"x": 1, "y": 2, "reward": -100}, {"x": 2, "y": 1, "reward": -100}],
                "BOARD_GOAL": {"x": 2, "y": 2, "reward": 100},
                "UNIT": 100,
                "WIDTH": 6,
                "HEIGHT": 5
            }
        },
        "monte_carlo": {
            "BOARD_BLOCKERS": [{"x": 1, "y": 2, "reward": -100}, {"x": 2, "y": 1, "reward": -100}],
            "BOARD_GOAL": {"x": 2, "y": 2, "reward": 100},
            "UNIT": 100,
            "WIDTH": 6,
            "HEIGHT": 5
        },
        "sarsa": {
            "DISPLAY": {
                "BOARD_BLOCKERS": [{"x": 1, "y": 2, "reward": -100}, {"x": 2, "y": 1, "reward": -100}],
                "BOARD_GOAL": {"x": 2, "y": 2, "reward": 100},
                "UNIT": 100,
                "WIDTH": 6,
                "HEIGHT": 5
            }
        },
        "qlearn": {
            "BOARD_BLOCKERS": [{"x": 1, "y": 2, "reward": -100}, {"x": 2, "y": 1, "reward": -100}],
            "BOARD_GOAL": {"x": 2, "y": 2, "reward": 100},
            "UNIT": 100,
            "WIDTH": 6,
            "HEIGHT": 5
        }
    }
)