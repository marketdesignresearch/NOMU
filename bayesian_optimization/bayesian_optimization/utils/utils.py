import numpy as np


def get_grid_points_multidimensional(lower_bounds, upper_bounds, n_gridpoints) -> np.array:
    """generate the grid-points for which the function is evaluated and for which the maximum evaluated.
    :return: grid-points as a mesh (can be multidimensional)
    """
    grid_res = np.meshgrid(
        *[np.linspace(low, upper_bounds[index], n_gridpoints) for (index, low) in
          enumerate(lower_bounds)])
    return np.array(list(zip(*np.vstack(list(map(np.ravel, grid_res))))))


def get_grid_points_multidimensional_and_grid(lower_bounds, upper_bounds, n_gridpoints) -> np.array:
    """generate the grid-points for which the function is evaluated and for which the maximum evaluated.
    :return: grid-points as a mesh (can be multidimensional)
    """
    grid_res = np.meshgrid(
        *[np.linspace(low, upper_bounds[index], n_gridpoints) for (index, low) in
          enumerate(lower_bounds)])
    return np.array(list(zip(*np.vstack(list(map(np.ravel, grid_res)))))), grid_res


def read_list_of_int_from_config(string):
    return [int(item) for item in string.split(",")]

def read_list_of_float_from_config(string):
    return [float(item) for item in string.split(",")]

def config_int_or_none(input):
    if input == "":
        return None
    return int(input)

def config_float_or_none(input):
    if input == "":
        return None
    return float(input)

def config_string_or_none(input):
    if input == "":
        return None
    return str(input)

def config_list_int_or_none(config, key):
    if config[key] == "":
        return None
    return [int(i) for i in config.as_list(key)]

def config_list_float_or_none(config, key):
    if config[key] == "":
        return None
    return [float(i) for i in config.as_list(key)]

def config_multiplication_list(config, key):
    list = config.as_list(key)
    if len(list) == 3 and list[1] == "*":
        return [float(list[0])]*list[2]
    else:
        return [float(i) for i in config.as_list(key)]


