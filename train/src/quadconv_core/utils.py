'''
Miscellaneous utility functions.
'''

import os
import yaml

'''
'''
def load_model_config(config_path):
    try:
        #open YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

    except Exception as e:
        raise ValueError(f"Config {config_path} is invalid.")

    return config

'''
Package conv parameters.

Input:
    kwargs: keyword arguments
'''
def package_args(stages, kwargs, mirror=False):

    for key, value in kwargs.items():
        if len(value) == 1:
            kwargs[key] = value*(stages)
        elif mirror:
            value.reverse() #inplace

    arg_stack = [{ key : value[i] for key, value in kwargs.items() } for i in range(stages)]

    return arg_stack

'''
Swap input and output points and channels
'''
def swap(conv_params):
    swapped_params = conv_params.copy()

    try:
        temp = swapped_params["in_points"]
        swapped_params["in_points"] = swapped_params["out_points"]
        swapped_params["out_points"] = temp
    except:
        pass

    try:
        temp = swapped_params["in_channels"]
        swapped_params["in_channels"] = swapped_params["out_channels"]
        swapped_params["out_channels"] = temp
    except:
        pass

    return swapped_params
