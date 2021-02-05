from ws.RLUtils.common.app_info_lib import DotDict

app_info = DotDict(
    {
        "NUM_EPISODES": 500,
        "BATCH_SIZE": 64,
        "GAMMA": 0.99,
        "LEARNING_RATE": 0.01,
        "EPSILON": 0.1,
        "RHO": 0.99,
        "DISCOUNT_FACTOR": 0.9,
    }
)

def fn_add_configs(api_info):
    for k, v in app_info.items():
        api_info[k] = v


