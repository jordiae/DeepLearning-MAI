import os
from PIL import Image
from cnn.models import *


def dir_path(s):
    if os.path.isdir(s):
        return s
    else:
        raise NotADirectoryError(s)


def get_num_pixels(img_path):
    width, height = Image.open(img_path).size
    return width, height


def load_arch(args):
    if args.arch == 'BaseCNN':
        model = BaseCNN()
    elif args.arch == 'AlexNet':
        model = AlexNet()
    elif args.arch == 'FiveLayerCNN':
        model = FiveLayerCNN()
    elif args.arch == 'AlbertCNN':
        model = AlbertCNN()
    elif args.arch == 'PyramidCNN':
        model = PyramidCNN(args)
    else:
        raise NotImplementedError()
    return model
