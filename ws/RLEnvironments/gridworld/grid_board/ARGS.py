from ws.RLUtils.common.DotDict import DotDict

args = DotDict(
    {
        "policy_iterator": {
            "display": {
                "BOARD_BLOCKERS": [{"x": 1, "y": 2, "reward": -100}, {"x": 2, "y": 1, "reward": -100}],
                "BOARD_GOAL": {"x": 2, "y": 2, "reward": 100},
                "UNIT": 100,
                "WIDTH": 6,
                "HEIGHT": 5
            }
        },
        "value_iterator": {
            "display": {
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
            "display": {
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