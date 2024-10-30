import logging

import torch


def get_device_type() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        logging.warning("No GPU found, using CPU instead")
        return "cpu"

def get_box_center(box):
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y
