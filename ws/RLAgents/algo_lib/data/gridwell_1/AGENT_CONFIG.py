from ws.RLUtils.common.DotDict import DotDict

args = DotDict(
    {
        "POSSIBLE_ACTIONS": [0, 1, 2, 3] ,
        "INITIAL_ACTION_PROBABILITIES": [.25, .25, .25, .25],
        "INITIAL_ACTION_VALUES": [0, 0, 0, 0.0],
        "ACTION_MOVE_STATE_RULES": [(0, -1), (0, 1), (-1, 0), (1, 0)],
    }
)

def add_configs(api_info):
    for k, v in args.items():
        api_info[k] = v


