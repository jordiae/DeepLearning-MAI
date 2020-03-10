import os
from PIL import Image


def dir_path(s):
    if os.path.isdir(s):
        return s
    else:
        raise NotADirectoryError(s)


def get_num_pixels(img_path):
    width, height = Image.open(img_path).size
    return width, height
