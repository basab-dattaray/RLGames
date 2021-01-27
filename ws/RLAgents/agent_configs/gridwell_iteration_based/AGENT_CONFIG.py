from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
    {
        "DISCOUNT_FACTOR": 0.9,
    }
)

def fn_add_configs(api_info):
    for k, v in app_info.items():
        api_info[k] = v


