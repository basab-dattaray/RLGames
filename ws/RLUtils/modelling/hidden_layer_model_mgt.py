import importlib
import traceback

import torch.nn as nn
from ws.RLInterfaces.PARAM_KEY_NAMES import NUM_NODES, ACTIVATION_FN, LAYER_TYPE, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, \
    STRIDE, PADDING, NUM_FEATURES


def hidden_layer_model_mgr(model_instance, verbose=True):
    # default_activation_function_name = "tanh"
    activation_module = importlib.import_module("torch")

    # default_activation_fn = getattr(activation_module, default_activation_function_name)

    def fn_hidden_layers_input_proc(prev_hidden_layer_dim):
        layer_count = 1
        for current_hidden_layer in model_instance._hidden_layer_dims:
            layer_name = 'hidden' + str(layer_count)
            if (current_hidden_layer[LAYER_TYPE]) == 'LINEAR':
                current_hidden_layer_dim = current_hidden_layer[NUM_NODES]
                model_instance.__setattr__(layer_name, nn.Linear(prev_hidden_layer_dim, current_hidden_layer_dim))
            elif (current_hidden_layer[LAYER_TYPE]) == 'CONV':
                stride = 1 if STRIDE not in current_hidden_layer.keys() else current_hidden_layer[STRIDE]
                padding = 0 if PADDING not in current_hidden_layer.keys() else current_hidden_layer[PADDING]
                model_instance.__setattr__(layer_name, nn.Conv2d(
                    current_hidden_layer[IN_CHANNELS],
                    current_hidden_layer[OUT_CHANNELS],
                    current_hidden_layer[KERNEL_SIZE],
                    stride=stride,
                    padding=padding
                ))
            elif (current_hidden_layer[LAYER_TYPE]) == 'BATCHNORM1D':
                model_instance.__setattr__(layer_name, nn.BatchNorm1d(current_hidden_layer[NUM_FEATURES]
                                                                      ))
            elif (current_hidden_layer[LAYER_TYPE]) == 'BATCHNORM2D':
                model_instance.__setattr__(layer_name, nn.BatchNorm2d(current_hidden_layer[NUM_FEATURES]
                                                                      ))
            else:
                exit()
            layer_count += 1
            prev_hidden_layer_dim = current_hidden_layer_dim
        return prev_hidden_layer_dim

    def fn_hidden_layers_forward_proc(layer_data):
        try:
            layer_count = 1
            for current_hidden_layer in model_instance._hidden_layer_dims:
                # activation_function = default_activation_fn
                activation_function = None
                if ACTIVATION_FN in current_hidden_layer.keys():
                    activation_function_name = current_hidden_layer[ACTIVATION_FN]
                    if activation_function_name is not None:
                        activation_function = getattr(activation_module, activation_function_name)

                layer_attr_ref = model_instance.__getattr__('hidden' + str(layer_count))
                layer_attr = layer_attr_ref(layer_data)
                if activation_function is not None:
                    layer_data = activation_function(layer_attr)
                layer_count += 1
        except Exception as x:
            if verbose:
                print(x)
                print(traceback.format_exc())
            exit()
        return layer_data

    return fn_hidden_layers_input_proc, fn_hidden_layers_forward_proc
