import torch

# from ws.RLInterfaces.PARAM_KEY_NAMES import FORCE_CPU_USE


def get_device(api_info):
    if 'FORCE_CPU_USE' in api_info.keys():
        if api_info['FORCE_CPU_USE'] == 1:
            return 'cpu'
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
