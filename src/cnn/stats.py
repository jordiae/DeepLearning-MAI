from utils import get_num_pixels
import os


def get_stats(data, path):
    stats = {}
    for class_ in data:
        stats[class_] = {}
        for img in data[class_]:
            width, height = get_num_pixels(os.path.join(path, class_, img))
            stats[class_][img] = width, height
    return stats


def get_freqs(img_dict):
    sizes = {}
    for img in img_dict:
        size = img_dict[img]
        if size in sizes:
            sizes[size] += 1
        else:
            sizes[size] = 1
    return sizes


def get_stats_freqs(data, path):
    stats = get_stats(data, path)
    res = []
    for key in stats:
        res.append(f'{key}: total = {len(stats[key])}, size_freqs = {get_freqs(stats[key])}\n\n')
    return res
