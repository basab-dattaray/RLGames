from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
    {
        "NUM_EPISODES": 500,
        "BATCH_SIZE": 64,
        "GAMMA": 0.99,
        "LEARNING_RATE": 0.01,
        "EPSILON": 0.1,
        "RHO": 0.99,
        "DISCOUNT_FACTOR": 0.9,

        # "POSSIBLE_ACTIONS": [0, 1, 2, 3] ,
        # "INITIAL_ACTION_PROBABILITIES": [.25, .25, .25, .25],
        # "INITIAL_ACTION_VALUES": [0, 0, 0, 0.0],
        "ACTION_MOVE_STATE_RULES": [(0, -1), (0, 1), (-1, 0), (1, 0)],
    }
)

def fn_add_configs(api_info):
    for k, v in app_info.items():
        api_info[k] = v


