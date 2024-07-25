import torch
import numpy as np


def grid_abs(height, width):
    x_t = np.ones((height, 1)) @ (np.linspace(-1, 1, width).reshape(1, width))  # [height, width]
    y_t = np.linspace(-1, 1, height).reshape(height, 1) @ np.ones((1, width))  # [height, width]
    x_t = (x_t + 1) * 0.5 * (width - 1)  # 先到[0, 1]再到[0, width-1]
    y_t = (y_t + 1) * 0.5 * (height - 1)

    x_t_flatten = x_t.reshape(1, -1)
    y_t_flatten = y_t.reshape(1, -1)
    ones = np.ones_like(x_t_flatten)

    grid = np.concatenate((x_t_flatten, y_t_flatten, ones), axis=0)  # [3, height*width]

    return grid
